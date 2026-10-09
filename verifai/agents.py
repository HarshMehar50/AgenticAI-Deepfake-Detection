from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from PIL import Image, ImageFilter
from torch import Tensor
from torchvision.transforms import functional as vision

from verifai.models import Detector, ModelRegistry

SENSITIVITY_THRESHOLD = 0.12


@dataclass(frozen=True)
class QualityReport:
    width: int
    height: int
    blur_score: float
    label: str


@dataclass(frozen=True)
class Selection:
    name: str
    reason: str


@dataclass(frozen=True)
class AdversarialReport:
    instability: float
    label: str
    explanation: str


@dataclass(frozen=True)
class FeatureReport:
    dimensions: int
    mean_activation: float
    frame_variation: float


@dataclass(frozen=True)
class Decision:
    verdict: str
    confidence: float
    explanation: str


class InputQualityAgent:
    def analyze(self, image: Image.Image) -> QualityReport:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        laplacian = (
            -4 * gray
            + np.roll(gray, 1, axis=0)
            + np.roll(gray, -1, axis=0)
            + np.roll(gray, 1, axis=1)
            + np.roll(gray, -1, axis=1)
        )
        blur_score = float(np.var(laplacian))
        width, height = image.size
        low_quality = min(width, height) < 320 or blur_score < 75
        label = "low" if low_quality else "adequate"
        return QualityReport(width, height, blur_score, label)

    def aggregate(self, reports: list[QualityReport]) -> QualityReport:
        if not reports:
            raise ValueError("At least one frame is required for quality assessment.")
        blur_score = float(np.mean([report.blur_score for report in reports]))
        width = min(report.width for report in reports)
        height = min(report.height for report in reports)
        label = "low" if any(report.label == "low" for report in reports) else "adequate"
        return QualityReport(width, height, blur_score, label)

    @staticmethod
    def to_tensor(image: Image.Image, size: int) -> Tensor:
        image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        return vision.to_tensor(image)


class ModelSelectionAgent:
    def select(self, registry: ModelRegistry, quality: QualityReport) -> Selection:
        desired = "low_quality" if quality.label == "low" else "high_quality"
        candidates = {registry[name].profile: name for name in registry.names}
        if desired in candidates:
            return Selection(
                candidates[desired],
                f"{desired.replace('_', ' ')} profile selected from resolution and blur.",
            )
        if "balanced" in candidates:
            return Selection(
                candidates["balanced"],
                f"Balanced profile selected; no {desired.replace('_', ' ')} specialist is configured.",
            )
        return Selection(
            registry.names[0],
            "Only one compatible detector is configured; it is used as the fallback.",
        )


class FeatureExtractionAgent:
    def analyze(self, images: list[Image.Image], detector: Detector) -> FeatureReport:
        if not images:
            raise ValueError("At least one frame is required for feature extraction.")
        tensors = torch.stack(
            [InputQualityAgent.to_tensor(image, 224) for image in images]
        ).to(detector.device)
        with torch.inference_mode():
            features = detector.extract_features(tensors)
        variation = (
            float(features.std(dim=0, unbiased=False).mean().item())
            if len(images) > 1
            else 0.0
        )
        return FeatureReport(
            dimensions=int(features.shape[1]),
            mean_activation=float(features.mean().item()),
            frame_variation=variation,
        )


class AdversarialAssessmentAgent:
    def __init__(
        self, instability_threshold: float = SENSITIVITY_THRESHOLD
    ) -> None:
        self.instability_threshold = instability_threshold

    def assess(
        self, images: list[Image.Image], detector: Detector, quality_agent: InputQualityAgent
    ) -> AdversarialReport:
        shifts: list[float] = []
        for image in images:
            tensor = quality_agent.to_tensor(image, 224).unsqueeze(0).to(detector.device)
            with torch.inference_mode():
                baseline = detector.logits(tensor).softmax(dim=1)[0, 1].item()
                brightness_probe = (tensor * 1.015).clamp(0, 1)
                blurred = image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=0.25))
                blur_probe = quality_agent.to_tensor(blurred, 224).unsqueeze(0).to(detector.device)
                brightness_probability = detector.logits(brightness_probe).softmax(dim=1)[0, 1].item()
                blur_probability = detector.logits(blur_probe).softmax(dim=1)[0, 1].item()
            shifts.extend(
                [abs(float(baseline - brightness_probability)), abs(float(baseline - blur_probability))]
            )
        instability = max(shifts, default=0.0)
        if instability >= self.instability_threshold:
            return AdversarialReport(
                instability,
                "High sensitivity",
                "Small benign image changes shifted the score substantially. This is a robustness "
                "warning, not proof that the upload is adversarial.",
            )
        return AdversarialReport(
            instability,
            "No high sensitivity observed",
            "The detector was stable under the tested benign changes; this does not rule out "
            "adversarial manipulation.",
        )


class DecisionVerificationAgent:
    def __init__(
        self,
        inconclusive_band: tuple[float, float] = (0.25, 0.75),
        instability_threshold: float = SENSITIVITY_THRESHOLD,
    ) -> None:
        self.inconclusive_band = inconclusive_band
        self.instability_threshold = instability_threshold

    def verify(self, probabilities: dict[str, float], instability: float) -> Decision:
        if not probabilities:
            raise ValueError("At least one detector result is required.")
        values = list(probabilities.values())
        labels = {probability >= 0.5 for probability in values}
        if len(labels) > 1:
            return Decision(
                "Inconclusive",
                0.0,
                "Configured detectors disagree, so the evidence does not support a verified verdict.",
            )
        probability = float(np.mean(values))
        low, high = self.inconclusive_band
        if low < probability < high:
            return Decision(
                "Inconclusive",
                0.0,
                "The detector score is too close to its decision boundary for a reliable verdict.",
            )
        if instability >= self.instability_threshold:
            return Decision(
                "Inconclusive",
                0.0,
                "The model is sensitive to benign input changes; its prediction is not verified.",
            )
        confidence = min(1.0, 2.0 * abs(probability - 0.5))
        if probability >= high:
            return Decision(
                "Likely deepfake",
                confidence,
                "The configured detectors consistently scored the media as likely manipulated.",
            )
        return Decision(
            "Likely authentic",
            confidence,
            "The configured detectors consistently scored the media as likely authentic.",
        )

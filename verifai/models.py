from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch
from torch import Tensor, nn
from torchvision.models import resnet18

PROFILES = {"balanced", "high_quality", "low_quality", "general"}


class DeepfakeClassifier(nn.Module):
    """ResNet-18 binary classifier; inputs are RGB tensors in [0, 1]."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def extract_features(self, images: Tensor) -> Tensor:
        images = (images - self.mean) / self.std
        backbone = self.backbone
        features = backbone.conv1(images)
        features = backbone.bn1(features)
        features = backbone.relu(features)
        features = backbone.maxpool(features)
        features = backbone.layer1(features)
        features = backbone.layer2(features)
        features = backbone.layer3(features)
        features = backbone.layer4(features)
        return torch.flatten(backbone.avgpool(features), 1)

    def forward(self, images: Tensor) -> Tensor:
        return self.backbone.fc(self.extract_features(images))


class Detector:
    def __init__(self, name: str, profile: str, model: DeepfakeClassifier) -> None:
        self.name = name
        self.profile = profile
        self.model = model.eval()
        self.device = next(model.parameters()).device

    def logits(self, images: Tensor) -> Tensor:
        return self.model(images.to(self.device))

    def extract_features(self, images: Tensor) -> Tensor:
        return self.model.extract_features(images.to(self.device))

    def fake_probability(self, image: Tensor) -> float:
        with torch.inference_mode():
            return float(self.logits(image.unsqueeze(0)).softmax(dim=1)[0, 1].item())


class ModelRegistry:
    def __init__(self, detectors: list[Detector]) -> None:
        if not detectors:
            raise ValueError("No detector checkpoints were loaded.")
        self._detectors = {detector.name: detector for detector in detectors}
        if len(self._detectors) != len(detectors):
            raise ValueError("Detector names must be unique.")

    @classmethod
    def from_directory(cls, directory: Path) -> ModelRegistry:
        paths = sorted(directory.glob("*.pt"))
        if not paths:
            raise ValueError(
                f"No .pt detector checkpoints found in {directory}. "
                "Train one with train.py first."
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        detectors: list[Detector] = []
        for path in paths:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict):
                raise ValueError(f"{path} is not a VERIFAI model checkpoint.")
            if checkpoint.get("architecture") != "resnet18":
                raise ValueError(f"Unsupported architecture in {path}.")
            if checkpoint.get("class_order") != ["real", "fake"]:
                raise ValueError(f"{path} must use class order ['real', 'fake'].")
            if not isinstance(checkpoint.get("state_dict"), dict):
                raise ValueError(f"{path} does not contain a model state dictionary.")
            profile = checkpoint.get("profile")
            if profile not in PROFILES:
                raise ValueError(f"{path} has an unsupported detector profile.")
            model = DeepfakeClassifier()
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.to(device).eval()
            detectors.append(Detector(path.stem, profile, model))
        return cls(detectors)

    @property
    def names(self) -> list[str]:
        return list(self._detectors)

    def items(self) -> Iterator[tuple[str, Detector]]:
        return iter(self._detectors.items())

    def __getitem__(self, name: str) -> Detector:
        return self._detectors[name]

    def __len__(self) -> int:
        return len(self._detectors)

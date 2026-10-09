from __future__ import annotations

import os
import tempfile
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from verifai.agents import (
    AdversarialAssessmentAgent,
    DecisionVerificationAgent,
    FeatureExtractionAgent,
    InputQualityAgent,
    ModelSelectionAgent,
)
from verifai.models import ModelRegistry


st.set_page_config(page_title="VERIFAI", layout="centered")
st.title("VERIFAI")
st.subheader("Adaptive Deepfake Detection & Adversarial Robustness")
st.caption(
    "Research prototype. Results depend on the model and data used to train it; "
    "they are not proof of authenticity."
)

MODEL_DIR = Path(os.environ.get("VERIFAI_MODELS_DIR", "models"))
MAX_VIDEO_FRAMES = 8
INPUT_SIZE = 224


def load_media(upload) -> tuple[list[Image.Image], str]:
    content = upload.getvalue()
    if upload.type.startswith("image/"):
        with Image.open(BytesIO(content)) as image:
            return [image.convert("RGB")], "image"

    suffix = Path(upload.name).suffix or ".mp4"
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temporary.write(content)
            temporary_path = temporary.name
        capture = cv2.VideoCapture(temporary_path)
        try:
            if not capture.isOpened():
                raise ValueError("Could not open this video. Check that its codec is supported.")
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                raise ValueError("The uploaded video contains no readable frames.")
            indices = np.linspace(
                0, frame_count - 1, min(MAX_VIDEO_FRAMES, frame_count), dtype=int
            )
            frames: list[Image.Image] = []
            for index in np.unique(indices):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if ok:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        finally:
            capture.release()
        if not frames:
            raise ValueError("No video frames could be decoded.")
        return frames, "video"
    finally:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)


@st.cache_resource
def get_registry(directory: str) -> ModelRegistry:
    return ModelRegistry.from_directory(Path(directory))


upload = st.file_uploader(
    "Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "webm"]
)

if not MODEL_DIR.exists() or not any(MODEL_DIR.glob("*.pt")):
    st.warning(
        f"No trained detector checkpoints found in `{MODEL_DIR}`. Train one with "
        "`python train.py --train-dir data/train --val-dir data/val --output models/balanced.pt` "
        "before requesting a verdict."
    )
    st.code("python train.py --help\npython evaluate.py --help", language="powershell")

if upload is not None:
    try:
        frames, media_kind = load_media(upload)
        if media_kind == "image":
            st.image(frames[0], caption="Uploaded image", use_container_width=True)
        else:
            st.video(upload.getvalue())
            st.caption(f"Analyzing {len(frames)} evenly sampled frames.")

        registry = get_registry(str(MODEL_DIR.resolve()))
        quality_agent = InputQualityAgent()
        quality_reports = [quality_agent.analyze(frame) for frame in frames]
        quality = quality_agent.aggregate(quality_reports)
        selector = ModelSelectionAgent()
        selection = selector.select(registry, quality)
        feature_report = FeatureExtractionAgent().analyze(frames, registry[selection.name])
        all_probabilities: dict[str, list[float]] = {name: [] for name in registry.names}

        with st.spinner(f"Running `{selection.name}` and verification detectors..."):
            for frame in frames:
                tensor = quality_agent.to_tensor(frame, INPUT_SIZE)
                for name, detector in registry.items():
                    probability = detector.fake_probability(tensor)
                    all_probabilities[name].append(probability)

        st.subheader("Agent findings")
        st.write(
            f"**Input quality:** {quality.label} "
            f"({quality.width}×{quality.height}, blur score {quality.blur_score:.1f})"
        )
        st.write(
            f"**Adaptive selection:** `{selection.name}` — {selection.reason}"
        )
        st.write(
            f"**Feature extraction:** {feature_report.dimensions}-dimensional embeddings; "
            f"mean activation {feature_report.mean_activation:.3f}, "
            f"frame variation {feature_report.frame_variation:.3f}"
        )

        mean_probabilities = {
            name: float(np.mean(values)) for name, values in all_probabilities.items()
        }
        adversarial_report = AdversarialAssessmentAgent().assess(
            frames, registry[selection.name], quality_agent
        )
        decision = DecisionVerificationAgent().verify(
            mean_probabilities, adversarial_report.instability
        )
        st.write(
            f"**Adversarial sensitivity:** {adversarial_report.label} "
            f"(maximum probe shift {adversarial_report.instability:.1%})"
        )
        st.caption(adversarial_report.explanation)

        st.subheader("Detector scores")
        for name, probability in mean_probabilities.items():
            st.write(f"`{name}` — fake probability: **{probability:.1%}**")

        st.subheader("Verified decision")
        if decision.verdict == "Inconclusive":
            st.info(decision.explanation)
        elif decision.verdict == "Likely deepfake":
            st.warning(f"{decision.verdict} · confidence {decision.confidence:.1%}")
            st.write(decision.explanation)
        else:
            st.success(f"{decision.verdict} · confidence {decision.confidence:.1%}")
            st.write(decision.explanation)
    except (OSError, ValueError, RuntimeError) as error:
        st.error(f"Analysis could not be completed: {error}")

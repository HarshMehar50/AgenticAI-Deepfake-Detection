from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from verifai.attacks import fgsm, pgd
from verifai.models import DeepfakeClassifier

INPUT_SIZE = 224


class Metrics(TypedDict):
    samples: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    roc_auc: float | None
    accuracy_drop_from_clean: NotRequired[float]


def metrics(labels: list[int], scores: list[float]) -> Metrics:
    predictions = [int(score >= 0.5) for score in scores]
    tp = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    tn = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    positives, negatives = tp + fn, tn + fp
    auc = _roc_auc(labels, scores)
    return {
        "samples": len(labels),
        "accuracy": (tp + tn) / max(1, len(labels)),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, positives),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "false_positive_rate": fp / max(1, negatives),
        "roc_auc": auc if math.isfinite(auc) else None,
    }


def _roc_auc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")
    ordered = sorted(zip(scores, labels))
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure clean, FGSM, and PGD performance on a labeled held-out dataset."
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder with fake/ and real/ subfolders.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.005, 0.01, 0.03])
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1 or args.pgd_steps < 1:
        parser.error("batch size and PGD steps must be positive.")
    if any(epsilon < 0 or epsilon > 1 for epsilon in args.epsilons):
        parser.error("epsilon values must be between 0 and 1.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dataset = datasets.ImageFolder(
        args.data_dir,
        transform=transforms.Compose(
            [transforms.Resize((INPUT_SIZE, INPUT_SIZE)), transforms.ToTensor()]
        ),
    )
    if set(dataset.classes) != {"fake", "real"}:
        raise ValueError("Evaluation data must contain exactly `fake` and `real` folders.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "resnet18" or checkpoint.get("class_order") != [
        "real",
        "fake",
    ]:
        raise ValueError("Unsupported or incompatible VERIFAI checkpoint.")
    model = DeepfakeClassifier().to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    fake_index = dataset.class_to_idx["fake"]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    result: dict[str, Metrics] = {}

    labels_all: list[int] = []
    clean_scores: list[float] = []
    for images, original_labels in loader:
        images = images.to(device)
        labels = (original_labels == fake_index).long().to(device)
        with torch.inference_mode():
            scores = model(images).softmax(dim=1)[:, 1]
        labels_all.extend(labels.cpu().tolist())
        clean_scores.extend(scores.cpu().tolist())
    result["clean"] = metrics(labels_all, clean_scores)

    for attack_name in ("fgsm", "pgd"):
        for epsilon in args.epsilons:
            labels_all = []
            attack_scores = []
            for images, original_labels in loader:
                images = images.to(device)
                labels = (original_labels == fake_index).long().to(device)
                if attack_name == "fgsm":
                    attacked = fgsm(model, images, labels, epsilon)
                else:
                    attacked = pgd(
                        model,
                        images,
                        labels,
                        epsilon,
                        step_size=max(epsilon / 4, 1 / 255),
                        steps=args.pgd_steps,
                    )
                with torch.inference_mode():
                    scores = model(attacked).softmax(dim=1)[:, 1]
                labels_all.extend(labels.cpu().tolist())
                attack_scores.extend(scores.cpu().tolist())
            result[f"{attack_name}_epsilon_{epsilon:g}"] = metrics(
                labels_all, attack_scores
            )
    clean_accuracy = result["clean"]["accuracy"]
    for key, value in result.items():
        if key != "clean":
            value["accuracy_drop_from_clean"] = clean_accuracy - value["accuracy"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

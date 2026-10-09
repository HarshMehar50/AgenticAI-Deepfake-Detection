from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18

from verifai.models import DeepfakeClassifier, PROFILES

INPUT_SIZE = 224


def make_dataset(
    directory: Path, training: bool, profile: str
) -> datasets.ImageFolder:
    operations: list[transforms.Transform] = [transforms.Resize((INPUT_SIZE, INPUT_SIZE))]
    if training:
        operations.append(transforms.RandomHorizontalFlip())
        if profile in {"balanced", "low_quality"}:
            operations.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=5)], p=0.25
                )
            )
        if profile == "low_quality":
            operations.append(transforms.RandomPosterize(bits=5, p=0.2))
    operations.extend([transforms.ToTensor()])
    dataset = datasets.ImageFolder(directory, transform=transforms.Compose(operations))
    if set(dataset.classes) != {"fake", "real"}:
        raise ValueError(f"{directory} must contain exactly `fake` and `real` folders.")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a VERIFAI binary detector from labeled real/fake images."
    )
    parser.add_argument("--train-dir", type=Path, required=True, help="Folder with fake/ and real/ subfolders.")
    parser.add_argument("--val-dir", type=Path, required=True, help="Held-out validation folder with fake/ and real/.")
    parser.add_argument("--output", type=Path, required=True, help="Checkpoint path, e.g. models/general.pt.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="balanced")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Initialize from ImageNet ResNet-18 weights (downloads weights if not cached).",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch size, and learning rate must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_data = make_dataset(args.train_dir, training=True, profile=args.profile)
    val_data = make_dataset(args.val_dir, training=False, profile=args.profile)
    if train_data.class_to_idx != val_data.class_to_idx:
        raise ValueError("Training and validation class mappings do not match.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    model = DeepfakeClassifier()
    if args.pretrained:
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Linear(backbone.fc.in_features, 2)
        model.backbone.load_state_dict(backbone.state_dict())
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    fake_index = train_data.class_to_idx["fake"]

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, original_labels in train_loader:
            labels = (original_labels == fake_index).long().to(device)
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * images.size(0)

        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for images, original_labels in val_loader:
                labels = (original_labels == fake_index).long().to(device)
                predictions = model(images.to(device)).argmax(dim=1)
                correct += int((predictions == labels).sum().item())
                total += labels.numel()
        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"train_loss={total_loss / len(train_data):.4f}, "
            f"val_accuracy={correct / total:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "resnet18",
            "class_order": ["real", "fake"],
            "profile": args.profile,
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "train_classes": train_data.class_to_idx,
            "epochs": args.epochs,
        },
        args.output,
    )
    print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()

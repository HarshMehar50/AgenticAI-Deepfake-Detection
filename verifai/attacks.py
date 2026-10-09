from __future__ import annotations

import torch
from torch import Tensor, nn


def fgsm(model: nn.Module, images: Tensor, labels: Tensor, epsilon: float) -> Tensor:
    """Create an L-infinity FGSM sample from [0, 1] RGB inputs."""
    _validate_epsilon(epsilon)
    inputs = images.detach().clone().requires_grad_(True)
    loss = nn.functional.cross_entropy(model(inputs), labels)
    gradient = torch.autograd.grad(loss, inputs)[0]
    return (inputs + epsilon * gradient.sign()).clamp(0, 1).detach()


def pgd(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    epsilon: float,
    step_size: float,
    steps: int,
    random_start: bool = True,
) -> Tensor:
    """Create an L-infinity PGD sample from [0, 1] RGB inputs."""
    _validate_epsilon(epsilon)
    if step_size <= 0 or steps < 1:
        raise ValueError("PGD step size and step count must be positive.")
    original = images.detach()
    if random_start:
        adversarial = (
            original + torch.empty_like(original).uniform_(-epsilon, epsilon)
        ).clamp(0, 1)
    else:
        adversarial = original.clone()
    for _ in range(steps):
        adversarial.requires_grad_(True)
        loss = nn.functional.cross_entropy(model(adversarial), labels)
        gradient = torch.autograd.grad(loss, adversarial)[0]
        adversarial = adversarial.detach() + step_size * gradient.sign()
        adversarial = torch.maximum(
            torch.minimum(adversarial, original + epsilon), original - epsilon
        ).clamp(0, 1)
    return adversarial.detach()


def _validate_epsilon(epsilon: float) -> None:
    if not 0 <= epsilon <= 1:
        raise ValueError("Epsilon must be between 0 and 1 for inputs in [0, 1].")

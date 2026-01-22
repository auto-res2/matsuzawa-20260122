from __future__ import annotations

"""Model factory and AALCR++ wrapper implementations."""

import torch.nn as nn
from torchvision.models import resnet18, resnet34

################################################################################
# CIFAR-adapted ResNet                                                         #
################################################################################

def _resnet_cifar(model_fn, num_classes: int = 10) -> nn.Module:
    m = model_fn(weights=None)
    # Adapt first layers for 32×32 resolution ---------------------------------
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

################################################################################
# AALCR++ wrapper                                                              #
################################################################################

class AALCRWrapper(nn.Module):
    """Adds a lightweight confidence branch to a classifier."""

    def __init__(self, base: nn.Module, num_classes: int):
        super().__init__()
        self.base = base
        self.aux_branch = nn.Sequential(
            nn.Linear(num_classes, num_classes // 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_classes // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, *, return_confidence: bool = False):
        logits = self.base(x)
        if return_confidence:
            confidence = self.aux_branch(logits).squeeze(1)  # shape: (B,)
            return logits, confidence
        return logits

################################################################################
# Public factory                                                               #
################################################################################

def create_model(cfg, *, num_classes: int):
    name = str(cfg.model.name).lower()
    if name == "resnet18":
        base = _resnet_cifar(resnet18, num_classes)
    elif name == "resnet34":
        base = _resnet_cifar(resnet34, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {name}")

    # Wrap with AALCR++ if required -------------------------------------------
    if str(cfg.method).startswith("AALCR"):
        return AALCRWrapper(base, num_classes)
    return base

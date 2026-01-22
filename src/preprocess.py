from __future__ import annotations

"""Data loading & preprocessing utilities for CIFAR-10 / CIFAR-100.
Returns DataLoaders that *always* yield (orig_img, aug_img, label) tuples to
avoid unpacking errors downstream.
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

################################################################################
# Dataset with dual transforms                                                 #
################################################################################

class DualTransformDataset(Dataset):
    """Wraps a base torchvision Dataset and returns (orig_img, aug_img, label)."""

    def __init__(self, base_ds: Dataset, orig_tf: transforms.Compose, aug_tf: transforms.Compose):
        self.base_ds = base_ds
        self.orig_tf = orig_tf
        self.aug_tf = aug_tf

    def __len__(self) -> int:  # noqa: D401 – PEP 257 single-line style
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        img, label = self.base_ds[idx]
        return self.orig_tf(img), self.aug_tf(img), label

################################################################################
# Public factory                                                               #
################################################################################

def get_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    name = str(cfg.dataset.name).lower()
    assert name in {"cifar10", "cifar100"}, f"Unsupported dataset: {name}"

    mean, std = cfg.dataset.preprocessing.normalize.mean, cfg.dataset.preprocessing.normalize.std
    norm = transforms.Normalize(mean=mean, std=std)

    orig_tf = transforms.Compose([
        transforms.ToTensor(),
        norm,
    ])
    aug_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        norm,
    ])

    root = Path(".cache") / name
    base_cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100

    # Load full training set (no transform – transforms handled in wrapper)
    full_train = base_cls(root=str(root), train=True, download=True)
    test_base = base_cls(root=str(root), train=False, download=True)

    # Split train/val ----------------------------------------------------------
    train_ratio = float(cfg.dataset.split.train)
    val_ratio = float(cfg.dataset.split.val)
    assert abs(train_ratio + val_ratio - 1.0) < 1e-6, "Train/val ratios must sum to 1"

    n_total = len(full_train)
    n_train = int(train_ratio * n_total)
    n_val = n_total - n_train

    g = torch.Generator().manual_seed(int(cfg.training.seed))
    train_base, val_base = random_split(full_train, [n_train, n_val], generator=g)

    # Wrap with dual-transform datasets ---------------------------------------
    train_ds = DualTransformDataset(train_base, orig_tf, aug_tf)
    val_ds = DualTransformDataset(val_base, orig_tf, orig_tf)   # no augmentation for val
    test_ds = DualTransformDataset(test_base, orig_tf, orig_tf)  # consistent tuple structure

    def _loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=int(cfg.training.batch_size),
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
        )

    return _loader(train_ds, True), _loader(val_ds, False), _loader(test_ds, False)

################################################################################
# Calibration metric                                                           #
################################################################################

def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error implementation (vectorised)."""
    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    confidences, _ = probs.max(dim=1)
    accuracies = (probs.argmax(1) == labels).float()

    ece = torch.tensor(0.0, device=probs.device)
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.float().mean() * torch.abs(accuracies[mask].mean() - confidences[mask].mean())
    return ece.item()

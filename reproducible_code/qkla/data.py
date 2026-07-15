"""Datasets + IDENTICAL augmentation across variants (protocol step 3).

Augmentation is a deliberate part of the fairness story: every model in a sweep
sees the byte-identical pipeline, so any accuracy gap is the attention operator,
not a richer recipe. We use a timm-lite floor (RandomCrop + flip + RandAugment +
normalize, optional mixup) without depending on timm.

`--synthetic` gives an offline, deterministic toy task for smoke-testing the
harness without downloading anything; the real runs use CIFAR-100/10.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

# per-dataset channel stats
_STATS = {
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "cifar10":  ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}
NUM_CLASSES = {"cifar100": 100, "cifar10": 10, "synthetic": 10}


def _transforms(dataset, image_size, train, augment):
    mean, std = _STATS.get(dataset, ((0.5,) * 3, (0.5,) * 3))
    if train and augment:
        ops = [T.RandomResizedCrop(image_size, scale=(0.7, 1.0), antialias=True),
               T.RandomHorizontalFlip(),
               T.RandAugment(num_ops=2, magnitude=9),
               T.ToTensor(), T.Normalize(mean, std)]
    else:
        ops = [T.Resize(image_size, antialias=True), T.CenterCrop(image_size),
               T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(ops)


class _SyntheticDS(Dataset):
    """Deterministic toy task: label = quantised mean intensity. Learnable, no
    download -- exists purely to exercise the harness offline."""

    def __init__(self, n, image_size, num_classes=10, seed=0):
        self.n, self.image_size, self.k = n, image_size, num_classes
        self.seed = seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(self.seed * 100003 + i)
        x = torch.rand(3, self.image_size, self.image_size, generator=g)
        y = int((x.mean() * self.k).clamp(0, self.k - 1))
        return x, y


def build_loaders(dataset="cifar100", *, data_dir="./data", image_size=32,
                  batch_size=128, workers=4, augment=True, synthetic=False,
                  subset=0, download=True):
    """Return (train_loader, val_loader, num_classes). `subset>0` truncates both
    splits (fast dev). `synthetic=True` ignores `dataset` and runs offline."""
    if synthetic or dataset == "synthetic":
        k = NUM_CLASSES["synthetic"]
        tr = _SyntheticDS(subset or 1024, image_size, k, seed=1)
        va = _SyntheticDS((subset or 256) // 4 or 64, image_size, k, seed=2)
        nc = k
    else:
        import torchvision.datasets as D
        ctor = {"cifar100": D.CIFAR100, "cifar10": D.CIFAR10}[dataset]
        tr = ctor(data_dir, train=True, download=download,
                  transform=_transforms(dataset, image_size, True, augment))
        va = ctor(data_dir, train=False, download=download,
                  transform=_transforms(dataset, image_size, False, augment))
        nc = NUM_CLASSES[dataset]
        if subset:
            tr = torch.utils.data.Subset(tr, range(min(subset, len(tr))))
            va = torch.utils.data.Subset(va, range(min(subset // 4 or 1, len(va))))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(tr, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(va, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=pin)
    return train_loader, val_loader, nc


def mixup(x, y, num_classes, alpha=0.2):
    """Standard mixup. Returns mixed inputs and SOFT targets (b, num_classes)."""
    if alpha <= 0:
        return x, torch.nn.functional.one_hot(y, num_classes).float()
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(x.size(0), device=x.device)
    x = lam * x + (1 - lam) * x[perm]
    oh = torch.nn.functional.one_hot(y, num_classes).float()
    return x, lam * oh + (1 - lam) * oh[perm]

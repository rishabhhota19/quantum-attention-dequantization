"""GPU-resident CIFAR-100 — the whole dataset lives in VRAM, batches are sampled
and augmented ON the GPU. No DataLoader, no worker processes, no CPU decode in
the loop -> the GPU never starves (≈100% util) and the Windows num_workers spawn
hang cannot happen. CIFAR-100 is ~154 MB as uint8, trivial in 4-48 GB.

Augmentation (reflect-pad + per-image random crop + random h-flip + normalize)
is done with tensor ops on-device, identical distribution to the torchvision
transforms it replaces.
"""

from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F
import pyarrow.parquet as pq
from PIL import Image

MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)


def _load_uint8(path):
    tb = pq.read_table(path)
    imgs = tb.column("img").to_pylist()
    arr = np.stack([np.asarray(Image.open(io.BytesIO(b["bytes"])).convert("RGB"),
                               dtype=np.uint8) for b in imgs])      # (N,32,32,3)
    lab = np.asarray(tb.column("fine_label").to_pylist(), dtype=np.int64)
    return arr, lab


class GPUCIFAR:
    """Iterable of (x, y) batches, everything resident on `device`."""

    def __init__(self, path, device, train, batch_size, drop_last=False):
        arr, lab = _load_uint8(path)
        self.x = torch.from_numpy(arr).to(device).permute(0, 3, 1, 2).contiguous()  # (N,3,32,32) uint8
        self.y = torch.from_numpy(lab).to(device)
        self.n = self.x.shape[0]
        self.bs, self.train, self.dev, self.drop_last = batch_size, train, device, drop_last
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    def __len__(self):
        return self.n // self.bs if self.drop_last else (self.n + self.bs - 1) // self.bs

    def _augment(self, xb):                                    # xb: uint8 (B,3,32,32)
        xb = xb.float().div_(255.0)
        if self.train:
            xb = F.pad(xb, (4, 4, 4, 4), mode="reflect")      # (B,3,40,40)
            b = xb.shape[0]
            top = torch.randint(0, 9, (b,), device=self.dev)
            left = torch.randint(0, 9, (b,), device=self.dev)
            ar = torch.arange(32, device=self.dev)
            rows = top.view(b, 1, 1) + ar.view(1, 32, 1)      # (B,32,1)
            cols = left.view(b, 1, 1) + ar.view(1, 1, 32)     # (B,1,32)
            bidx = torch.arange(b, device=self.dev).view(b, 1, 1)
            xb = xb[bidx, :, rows, cols].permute(0, 3, 1, 2)  # (B,3,32,32) cropped
            flip = torch.rand(b, device=self.dev) < 0.5
            xb[flip] = torch.flip(xb[flip], dims=[-1])
        return (xb - self.mean) / self.std

    def __iter__(self):
        idx = (torch.randperm(self.n, device=self.dev) if self.train
               else torch.arange(self.n, device=self.dev))
        last = self.n - (self.n % self.bs) if self.drop_last else self.n
        for i in range(0, last, self.bs):
            j = idx[i:i + self.bs]
            yield self._augment(self.x[j]), self.y[j]

"""Real CIFAR-100, 1-epoch sanity on the local GPU, HF-ViT backbone.

Uses the HuggingFace parquet (fast CDN) instead of torchvision's slow source.
Purpose: confirm on REAL data (not synthetic) that the trusted-baseline softmax
and our cos-feature variants all learn and stay stable in one epoch, before
committing the L40. Numbers at 1 epoch are indicative only.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import pyarrow.parquet as pq
from PIL import Image

from qkla.hf_vit import build_hf_vit, count_params
from qkla.engine import set_seed

DATA = Path("./data/hf_cifar100/cifar100")
MEAN, STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)


class ParquetCIFAR(Dataset):
    def __init__(self, path, train):
        import numpy as np
        tb = pq.read_table(path)
        imgs = tb.column("img").to_pylist()               # PNG bytes (heavy, transient)
        self.lab = tb.column("fine_label").to_pylist()
        # Decode ALL PNGs ONCE into a compact uint8 array (~150MB for 50k vs ~2GB of
        # Python PNG-byte objects), then drop the bytes. Removes per-batch decode and
        # slashes system-RAM use -- important on a 4GB-VRAM / 16GB-RAM laptop.
        self.data = np.stack([
            np.asarray(Image.open(io.BytesIO(b["bytes"])).convert("RGB"), dtype=np.uint8)
            for b in imgs])                               # (N, 32, 32, 3) uint8
        del imgs
        if train:
            self.tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                                 T.ToTensor(), T.Normalize(MEAN, STD)])
        else:
            self.tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, i):
        return self.tf(Image.fromarray(self.data[i])), self.lab[i]   # cheap, no decode


def main(variants=("softmax", "performer", "gaussian_rff", "quantum"), epochs=1, bs=128):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} {torch.cuda.get_device_name(0) if dev=='cuda' else ''}", flush=True)
    # num_workers=0: on Windows, workers spawn+pickle the whole in-RAM dataset
    # per worker -> thrash. Data is already in memory, so decode in-process.
    tr = DataLoader(ParquetCIFAR(DATA / "train-00000-of-00001.parquet", True),
                    batch_size=bs, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    va = DataLoader(ParquetCIFAR(DATA / "test-00000-of-00001.parquet", False),
                    batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
    cfg = dict(image_size=32, patch_size=4, num_classes=100, hidden_size=192,
               depth=6, heads=3, mlp_dim=384, num_features=256)

    rows = {}
    for v in variants:
        set_seed(0)
        m = build_hf_vit(v, **cfg).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=0.05)
        scaler = torch.cuda.amp.GradScaler() if dev == "cuda" else None
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for ep in range(epochs):
            m.train()
            for x, y in tr:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=dev.split(":")[0], enabled=scaler is not None):
                    loss = F.cross_entropy(m(pixel_values=x).logits, y, label_smoothing=0.1)
                if scaler:
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                else:
                    loss.backward(); opt.step()
        m.eval(); correct = seen = 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(dev), y.to(dev)
                correct += (m(pixel_values=x).logits.argmax(-1) == y).sum().item(); seen += x.size(0)
        acc = 100 * correct / seen
        peak = torch.cuda.max_memory_allocated() / 1e6 if dev == "cuda" else float("nan")
        rows[v] = (acc, count_params(m), (time.time() - t0) / epochs, peak)
        print(f"[real] {v:13s} acc={acc:5.2f}  params={count_params(m):,}  "
              f"{rows[v][2]:.0f}s/ep  peak={peak:.0f}MB", flush=True)

    print("\n=== REAL CIFAR-100, 1 epoch (chance=1.0%) ===", flush=True)
    print(f"{'variant':14s}{'top-1%':>8s}{'params':>12s}{'s/ep':>7s}{'peakMB':>8s}", flush=True)
    for v, (a, p, s, mem) in rows.items():
        print(f"{v:14s}{a:8.2f}{p:>12,}{s:7.0f}{mem:8.0f}", flush=True)
    if "quantum" in rows and "gaussian_rff" in rows:
        d = rows["quantum"][0] - rows["gaussian_rff"][0]
        print(f"\nquantum - gaussian_rff = {d:+.2f} pts (1-epoch, indicative)", flush=True)


if __name__ == "__main__":
    main()

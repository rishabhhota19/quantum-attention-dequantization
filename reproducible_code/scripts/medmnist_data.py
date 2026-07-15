"""GPU-resident MedMNIST (PathMNIST + DermaMNIST) — mirrors the CIFAR loader so
``scripts/p1_train_one.py`` can train on MedMNIST behind a ``--dataset`` flag,
with ZERO changes to the matched-budget / bf16 / on-GPU-augmentation rig.

Why MedMNIST: a second, out-of-domain benchmark (medical 2D images) for the
dequantization study — same harness, same matched 1,823,908-param ViT, different
data distribution. Two tasks:
    * PathMNIST  — 9-class colon-pathology tiles, 28x28 RGB,   ~107k images.
    * DermaMNIST — 7-class dermatoscopy (HAM10000), 28x28 RGB, ~10k images.
Both are natively 3-channel RGB, so the ViT's ``num_channels=3`` stem is reused
unchanged.

IMAGE-SIZE DECISION (documented, deliberate):
    MedMNIST tiles are 28x28. The publishable rig calls build_hf_vit with the
    DEFAULTS image_size=32, patch_size=4 (8x8=64 patches), identical to CIFAR-100.
    Rather than introduce a second ViT geometry (which would change the patch
    grid, the positional-embedding count and therefore the parameter count, and
    break ``assert_param_parity`` against the CIFAR reference of 1,823,908), we
    UPSAMPLE 28x28 -> 32x32 with bilinear interpolation on-GPU. This keeps
    build_hf_vit, the attention swap, and the param count byte-identical to the
    CIFAR runs (28 with patch 4 would give a 7x7=49 patch grid and a DIFFERENT
    pos-embed size -> a different, non-comparable parameter budget). Upsampling
    is done once per batch on the GPU (negligible) so the data path stays
    GPU-resident with ~100% util and no DataLoader / Windows-worker hang.

Contract (identical to qkla/gpu_data.GPUCIFAR):
    iterating a loader yields ``(x, y)`` where x is a normalised float tensor of
    shape (B, 3, 32, 32) on ``device`` and y is int64 labels on ``device``.
    ``p1_train_one`` feeds ``model(pixel_values=x).logits``.

Pod deps (note for the A40 pod):
    transformers==4.57.6  (v5 renames ViTSelfAttention -> breaks the swap),
    einops, performer_pytorch, pyarrow, pillow, huggingface_hub,
    medmnist  (pulls the npz; pip install medmnist).

GPU-resident, on-GPU augmentation, no DataLoader in the CUDA path — exactly like
GPUCIFAR. The CPU fallback (no CUDA) mirrors scripts.local_real_test.ParquetCIFAR
so a laptop sanity run still works (num_workers=0 on Windows).

Standalone smoke test (no GPU needed; downloads the npz the first time):
    python scripts/medmnist_data.py --dataset pathmnist
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Dataset registry. ``flag`` is the medmnist key; ``num_classes`` is fixed by
# the official split. MEAN/STD are the standard MedMNIST per-channel stats (the
# npz are uint8 in [0,255]; we normalise to roughly zero-mean/unit-var, mirroring
# the CIFAR MEAN/STD convention). MedMNIST images are mild-contrast medical tiles
# so a 0.5/0.5 normalisation is a safe, dataset-agnostic choice used across the
# MedMNIST baselines; kept identical for every attention variant -> still fair.
# ---------------------------------------------------------------------------
MEDMNIST_INFO = {
    "pathmnist":  {"flag": "pathmnist",  "num_classes": 9,
                   "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    "dermamnist": {"flag": "dermamnist", "num_classes": 7,
                   "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
}

# Native MedMNIST tile size and the ViT input size we upsample to (see header).
NATIVE_SIZE = 28
TARGET_SIZE = 32


def is_medmnist(dataset: str) -> bool:
    return dataset.lower() in MEDMNIST_INFO


def num_classes(dataset: str) -> int:
    return MEDMNIST_INFO[dataset.lower()]["num_classes"]


# ---------------------------------------------------------------------------
# npz loading. We use the medmnist pip package's npz (medmnist.<Class>(...)) so
# the download is the official, checksummed file. We pull raw uint8 arrays and
# do ALL tensor work ourselves (resident + on-GPU augment), exactly like the
# CIFAR loader pulls raw uint8 out of the parquet.
# ---------------------------------------------------------------------------
def _load_uint8(dataset: str, split: str, data_dir: str):
    """Return (imgs uint8 (N,28,28,3), labels int64 (N,)) for one split.

    Downloads via the medmnist package on first use (download=True). MedMNIST
    label arrays are shaped (N,1); we squeeze to (N,). PathMNIST/DermaMNIST are
    already 3-channel RGB so imgs come back (N,28,28,3); we still guard a
    grayscale shape just in case a flag is swapped in later.
    """
    info = MEDMNIST_INFO[dataset.lower()]
    import medmnist
    from medmnist import INFO

    cls_name = INFO[info["flag"]]["python_class"]      # e.g. "PathMNIST"
    DataClass = getattr(medmnist, cls_name)
    os.makedirs(data_dir, exist_ok=True)
    ds = DataClass(split=split, download=True, root=data_dir)

    imgs = np.asarray(ds.imgs)                          # (N,28,28[,3]) uint8
    if imgs.ndim == 3:                                  # grayscale -> 3ch
        imgs = np.repeat(imgs[..., None], 3, axis=-1)
    imgs = np.ascontiguousarray(imgs, dtype=np.uint8)
    lab = np.asarray(ds.labels, dtype=np.int64).reshape(-1)   # (N,1)->(N,)
    return imgs, lab


# ---------------------------------------------------------------------------
# GPU-resident loader — the CUDA path. Mirrors qkla.gpu_data.GPUCIFAR exactly:
# whole split lives in VRAM as uint8, batches sampled + augmented on-device. The
# only addition is the 28->32 bilinear upsample (see header).
# ---------------------------------------------------------------------------
class GPUMedMNIST:
    """Iterable of (x, y) batches, everything resident on ``device``.

    Same interface as GPUCIFAR: ``GPUMedMNIST(dataset, device, train, batch_size,
    drop_last=False, data_dir=...)``. x is float (B,3,32,32) normalised; y int64.
    """

    def __init__(self, dataset, device, train, batch_size, drop_last=False,
                 data_dir="./data/medmnist"):
        split = "train" if train else "test"
        arr, lab = _load_uint8(dataset, split, data_dir)
        info = MEDMNIST_INFO[dataset.lower()]
        self.x = torch.from_numpy(arr).to(device).permute(0, 3, 1, 2).contiguous()  # (N,3,28,28) uint8
        self.y = torch.from_numpy(lab).to(device)
        self.n = self.x.shape[0]
        self.bs, self.train, self.dev, self.drop_last = batch_size, train, device, drop_last
        self.mean = torch.tensor(info["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(info["std"], device=device).view(1, 3, 1, 1)

    def __len__(self):
        return self.n // self.bs if self.drop_last else (self.n + self.bs - 1) // self.bs

    def _augment(self, xb):                                    # xb: uint8 (B,3,28,28)
        xb = xb.float().div_(255.0)
        # upsample 28 -> 32 so the ViT geometry (image_size=32, patch 4) and the
        # 1,823,908-param budget match the CIFAR rig exactly (see header).
        xb = F.interpolate(xb, size=(TARGET_SIZE, TARGET_SIZE),
                           mode="bilinear", align_corners=False)
        if self.train:
            xb = F.pad(xb, (4, 4, 4, 4), mode="reflect")      # (B,3,40,40)
            b = xb.shape[0]
            top = torch.randint(0, 9, (b,), device=self.dev)
            left = torch.randint(0, 9, (b,), device=self.dev)
            ar = torch.arange(TARGET_SIZE, device=self.dev)
            rows = top.view(b, 1, 1) + ar.view(1, TARGET_SIZE, 1)     # (B,32,1)
            cols = left.view(b, 1, 1) + ar.view(1, 1, TARGET_SIZE)    # (B,1,32)
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


# ---------------------------------------------------------------------------
# CPU fallback — a torch Dataset mirroring scripts.local_real_test.ParquetCIFAR,
# so a no-CUDA laptop sanity run still works. The 28->32 upsample is folded into
# the transform; normalisation matches the GPU path. Decode is trivial (npz is
# already a uint8 array), so no per-item PNG decode.
# ---------------------------------------------------------------------------
class MedMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, train, data_dir="./data/medmnist"):
        import torchvision.transforms as T
        split = "train" if train else "test"
        self.data, self.lab = _load_uint8(dataset, split, data_dir)   # (N,28,28,3) uint8, (N,)
        info = MEDMNIST_INFO[dataset.lower()]
        from PIL import Image
        self._Image = Image
        norm = T.Normalize(info["mean"], info["std"])
        if train:                                         # Resize 28->32 then augment at 32
            self.tf = T.Compose([T.Resize(TARGET_SIZE),
                                 T.RandomCrop(TARGET_SIZE, padding=4),
                                 T.RandomHorizontalFlip(), T.ToTensor(), norm])
        else:
            self.tf = T.Compose([T.Resize(TARGET_SIZE), T.ToTensor(), norm])

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, i):
        img = self._Image.fromarray(self.data[i])         # (28,28,3) uint8 -> PIL
        return self.tf(img), int(self.lab[i])


# ---------------------------------------------------------------------------
# Convenience: build (train, val) loaders for either path, mirroring how
# p1_train_one selects GPUCIFAR vs DataLoader(ParquetCIFAR). Returns objects
# that yield the (pixel_values, labels) contract p1_train_one expects.
# ---------------------------------------------------------------------------
def build_loaders(dataset, device, batch_size, workers=0,
                  data_dir="./data/medmnist", val_batch_size=256):
    """Return (train_loader, val_loader) honouring the GPU-resident vs CPU split.

    On CUDA: GPU-resident GPUMedMNIST (no DataLoader, on-GPU augment).
    On CPU:  DataLoader(MedMNISTDataset) with num_workers=workers (use 0 on
             Windows to avoid the worker spawn hang).
    """
    if device == "cuda" or (isinstance(device, str) and device.startswith("cuda")):
        tr = GPUMedMNIST(dataset, device, True, batch_size, drop_last=True, data_dir=data_dir)
        va = GPUMedMNIST(dataset, device, False, val_batch_size, data_dir=data_dir)
        return tr, va
    from torch.utils.data import DataLoader
    tr = DataLoader(MedMNISTDataset(dataset, True, data_dir=data_dir),
                    batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    va = DataLoader(MedMNISTDataset(dataset, False, data_dir=data_dir),
                    batch_size=val_batch_size, num_workers=workers)
    return tr, va


def ensure_medmnist(dataset, data_dir="./data/medmnist"):
    """Idempotently fetch both splits of a MedMNIST flag via the pip package."""
    for split in ("train", "test"):
        _load_uint8(dataset, split, data_dir)
    return data_dir


if __name__ == "__main__":     # standalone smoke test (no GPU required)
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="pathmnist", choices=list(MEDMNIST_INFO))
    p.add_argument("--data-dir", default="./data/medmnist")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} dataset={args.dataset} "
          f"num_classes={num_classes(args.dataset)}", flush=True)
    tr, va = build_loaders(args.dataset, dev, batch_size=8, workers=0,
                           data_dir=args.data_dir)
    xb, yb = next(iter(tr))
    print(f"train batch x={tuple(xb.shape)} dtype={xb.dtype} "
          f"y={tuple(yb.shape)} y[:8]={yb[:8].tolist()}", flush=True)
    assert xb.shape[1:] == (3, TARGET_SIZE, TARGET_SIZE), xb.shape
    assert int(yb.max()) < num_classes(args.dataset)
    print("OK: (pixel_values, labels) contract holds", flush=True)

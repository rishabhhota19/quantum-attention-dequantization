"""P1 single run: train ONE (variant, seed) to convergence and checkpoint it.

The publishable rig: HF-ViT backbone (trusted softmax), matched params, bf16 by
default (fixes performer + iqp fp16 collapse, same precision for all variants ->
no confound), AdamW + cosine schedule w/ warmup, label smoothing. Writes the
full per-epoch curve + best_val_acc to results/p1/<tag>.json (resumable: if the
json exists and is complete, it is skipped).

Run by the orchestrator (scripts/run_p1.py); also usable standalone:
    python scripts/p1_train_one.py --variant quantum --layers 3 --seed 0 \
        --epochs 100 --precision bf16
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from qkla.hf_vit import build_hf_vit, count_params
from qkla.engine import set_seed


def ensure_cifar100(data_dir="./data"):
    """Idempotently fetch the CIFAR-100 parquet from the HF CDN (fast)."""
    base = Path(data_dir) / "hf_cifar100" / "cifar100"
    if (base / "train-00000-of-00001.parquet").exists():
        return base
    from huggingface_hub import hf_hub_download
    for split in ("train", "test"):
        hf_hub_download(repo_id="uoft-cs/cifar100", repo_type="dataset",
                        filename=f"cifar100/{split}-00000-of-00001.parquet",
                        local_dir=str(Path(data_dir) / "hf_cifar100"))
    return base


def _autocast(precision, device):
    if precision == "bf16":
        return torch.autocast(device_type=device, dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type=device, dtype=torch.float16)
    return contextlib.nullcontext()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--dataset", default="cifar100",
                   choices=["cifar100", "pathmnist", "dermamnist"],
                   help="cifar100 (default) or a MedMNIST flag (28x28 RGB, upsampled to 32)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--num-features", type=int, default=256)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--coupling", type=float, default=1.0)
    p.add_argument("--bandwidth", type=float, default=-1.0)   # <=0 -> auto
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="./results/p1")
    p.add_argument("--save-weights", action="store_true",
                   help="also save model.state_dict() to <tag>.pt (for E5 collapse-on-activations)")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    # default tag: keep CIFAR's historical form (run_p1.py filenames) but prefix
    # non-CIFAR datasets so MedMNIST runs land in distinct json checkpoints.
    _pfx = "" if args.dataset == "cifar100" else f"{args.dataset}_"
    tag = args.tag or f"{_pfx}{args.variant}_L{args.layers}_s{args.seed}"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{tag}.json"
    if path.exists() and json.loads(path.read_text()).get("done"):
        print(f"[{tag}] already complete -> skip", flush=True); return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU"
    print(f"[{tag}] device={dev} ({gpu_name}) precision={args.precision}", flush=True)
    set_seed(args.seed)
    if args.dataset == "cifar100":
        n_classes = 100
        ensure_cifar100(args.data_dir)
        from scripts.local_real_test import DATA
        if dev == "cuda":
            # whole dataset resident in VRAM, augment on-GPU -> ~100% util, no worker hang
            from qkla.gpu_data import GPUCIFAR
            tr = GPUCIFAR(DATA / "train-00000-of-00001.parquet", dev, True, args.batch_size, drop_last=True)
            va = GPUCIFAR(DATA / "test-00000-of-00001.parquet", dev, False, 256)
        else:
            from scripts.local_real_test import ParquetCIFAR
            tr = DataLoader(ParquetCIFAR(DATA / "train-00000-of-00001.parquet", True),
                            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
            va = DataLoader(ParquetCIFAR(DATA / "test-00000-of-00001.parquet", False),
                            batch_size=256, num_workers=args.workers)
    else:  # MedMNIST: same GPU-resident contract, 28x28 RGB upsampled to 32 (see medmnist_data header)
        from scripts.medmnist_data import build_loaders, num_classes as mm_classes
        n_classes = mm_classes(args.dataset)
        tr, va = build_loaders(args.dataset, dev, args.batch_size, workers=args.workers)

    bandwidth = None if args.bandwidth <= 0 else args.bandwidth
    model = build_hf_vit(args.variant, image_size=args.image_size, patch_size=args.patch_size,
                         num_classes=n_classes, num_features=args.num_features, layers=args.layers,
                         coupling=args.coupling, bandwidth=bandwidth).to(dev)
    params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warm, total = max(args.warmup, 0), max(args.epochs, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e + 1) / max(warm, 1) if e < warm
                                              else 0.5 * (1 + math.cos(math.pi * (e - warm) / max(total - warm, 1))))
    scaler = torch.cuda.amp.GradScaler() if (args.precision == "fp16" and dev == "cuda") else None
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history, best, t0 = [], 0.0, time.time()
    for ep in range(args.epochs):
        model.train(); te = time.time(); run = seen = 0.0
        for xb, yb in tr:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with _autocast(args.precision, dev):
                loss = F.cross_entropy(model(pixel_values=xb).logits, yb,
                                       label_smoothing=args.label_smoothing)
            if scaler:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            run += loss.item() * xb.size(0); seen += xb.size(0)
        sched.step()
        model.eval(); c = s = 0
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(dev), yb.to(dev)
                with _autocast(args.precision, dev):
                    c += (model(pixel_values=xb).logits.argmax(-1) == yb).sum().item()
                s += xb.size(0)
        acc = 100 * c / s; best = max(best, acc)
        history.append({"epoch": ep, "train_loss": run / seen, "val_acc": acc,
                        "epoch_sec": time.time() - te})
        print(f"[{tag}] ep{ep:3d} loss {run/seen:.3f} val {acc:5.2f} best {best:5.2f}", flush=True)

    peak = torch.cuda.max_memory_allocated() / 1e6 if dev == "cuda" else float("nan")
    import platform, datetime
    env = {"torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
           "host": platform.node(), "python": platform.python_version(),
           "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "bf16_supported": torch.cuda.is_bf16_supported() if dev == "cuda" else False}
    path.write_text(json.dumps({
        "tag": tag, "variant": args.variant, "seed": args.seed, "layers": args.layers,
        "coupling": args.coupling, "precision": args.precision, "params": params,
        "best_val_acc": best, "final_val_acc": history[-1]["val_acc"],
        "peak_mem_mb": peak, "wall_sec": time.time() - t0,
        "mean_epoch_sec": sum(h["epoch_sec"] for h in history) / len(history),
        "env": env, "config": vars(args), "history": history, "done": True}, indent=2))
    if args.save_weights:                       # for E5 (collapse on TRAINED activations)
        wpath = out / f"{tag}.pt"
        torch.save(model.state_dict(), wpath)
        print(f"[{tag}] saved weights -> {wpath}", flush=True)
    print(f"[{tag}] DONE best={best:.2f} -> {path}", flush=True)


if __name__ == "__main__":
    main()

"""Training / eval engine + the matched-budget instrumentation (protocol step 2).

Reports, per run: trainable params, FLOPs/forward, wall-clock/epoch, peak GPU
memory. FLOPs combine an exact nn.Linear MAC count (hooks) with an analytical
attention-core term -- the one part that is not a Linear and the one that
differs across variants (softmax O(n^2 d) vs linear O(n d r)).
"""

from __future__ import annotations

import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------- FLOPs
def _linear_macs(model, sample_input) -> int:
    """Exact multiply-accumulate count over all nn.Linear, via forward hooks."""
    total = 0
    handles = []

    def hook(mod, inp, out):
        nonlocal total
        total += inp[0].numel() * mod.out_features      # positions*in * out

    for m in model.modules():
        if isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(sample_input)
    if was_training:
        model.train()
    for h in handles:
        h.remove()
    return total


def _attention_core_macs(variant, *, n_tokens, depth, heads, dim_head, num_features) -> int:
    """MACs of the attention core op only (the matmuls/einsums that are not
    nn.Linear). softmax: 2 n^2 d per head; linear: ~4 n r d per head (phi(q),
    phi(k), kv-reduction, num)."""
    if variant == "softmax":
        per_head = 2 * n_tokens * n_tokens * dim_head
    else:
        per_head = 4 * n_tokens * num_features * dim_head
    return per_head * heads * depth


def profile_flops(model, sample_input, *, variant, n_tokens, depth, heads,
                  dim_head, num_features) -> float:
    """Total GFLOPs for one forward pass (FLOPs = 2 * MACs)."""
    macs = _linear_macs(model, sample_input)
    macs += _attention_core_macs(variant, n_tokens=n_tokens, depth=depth,
                                 heads=heads, dim_head=dim_head,
                                 num_features=num_features)
    return 2 * macs / 1e9


# ------------------------------------------------------------------- train/eval
def _loss(logits, target, label_smoothing):
    if target.dtype == torch.long:                      # hard labels
        return F.cross_entropy(logits, target, label_smoothing=label_smoothing)
    return -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()   # soft (mixup)


def train_one_epoch(model, loader, opt, device, *, scaler=None,
                    label_smoothing=0.0, mixup_fn=None, num_classes=None,
                    max_steps=0, grad_clip=0.0):
    model.train()
    running, seen, correct = 0.0, 0, 0
    for step, (x, y) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        target = mixup_fn(x, y, num_classes) if mixup_fn else y
        if isinstance(target, tuple):
            x, target = target
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.split(":")[0], enabled=scaler is not None):
            logits = model(x)
            loss = _loss(logits, target, label_smoothing)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        running += loss.item() * x.size(0); seen += x.size(0)
        correct += (logits.argmax(-1) == y).sum().item()
    return {"train_loss": running / max(seen, 1), "train_acc": correct / max(seen, 1)}


@torch.no_grad()
def evaluate(model, loader, device, *, max_steps=0):
    model.eval()
    correct, seen, loss_sum = 0, 0, 0.0
    for step, (x, y) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item(); seen += x.size(0)
    return {"val_loss": loss_sum / max(seen, 1), "val_acc": correct / max(seen, 1)}


def peak_mem_mb():
    return torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else float("nan")

"""E5 collapse diagnostic ON REAL TRAINED-MODEL ACTIVATIONS.

Companion to scripts/collapse_diag.py. That script computes the
effective-(entropy)-rank "spread shrinks under per-depth tuned bandwidth vs
fixed bandwidth" panel on the EXACT quantum kernel of SYNTHETIC unit-sphere
inputs. E5 recomputes the same panel on the REAL feature-map Gram
phi(q) phi(q)^T produced by the actual quantum-ViT, fed a real CIFAR-100 batch.

Why: collapse_diag shows the closed-form kernel collapses under tuning; E5 is
the confirmation that the SAME thing happens to the features the trained model
actually computes (real q activations -> real phi(q) -> real Gram), so the depth
U-curve really is a bandwidth artifact in vivo, not just in the idealised kernel.

What it does (mirrors collapse_diag structure):
  1. For each depth L in TUNED: build a depth-L quantum ViT, register a forward
     hook on every attention feature_map to capture phi(q) for a real batch,
     run ONE forward pass over a CIFAR-100 batch, pool the captured phi(q) over
     layers/heads, and compute the eff-rank of the real Gram phi(q) phi(q)^T at
     the TUNED bandwidth (expect ~MATCHED across L = collapse).
  2. Repeat at a FIXED bandwidth=1.0 (expect DIVERGENCE = the U-curve driver).
  3. Write results/p1/collapse_on_activations.json with tuned-vs-fixed eff-rank
     per depth, ready to drop in as a confirmation panel on the collapse figure.

NO TRAINING, but a real forward pass on a real batch. There is no trained-weight
checkpoint in this repo (p1 checkpoints store metrics+history, not weights), so
the model is freshly built per depth -- the quantum feature map is parameter-free
and the eff-rank of phi(q) phi(q)^T is dominated by the (bandwidth, layers) knob,
not by the trained q/k/v Linears (the U-curve claim is a property of the kernel,
which is what we measure). If a state_dict checkpoint is ever provided via
--ckpt, it is loaded so the q activations are the trained ones.

Pod deps: transformers==4.57.6 (v5 renames ViTSelfAttention), einops,
performer_pytorch, pyarrow, pillow, huggingface_hub, medmnist.

Precision: forward runs under bf16 autocast on CUDA (fp16 collapses the cos
features); eff-rank math is done in float64 on CPU for numerical stability.

Run (on the A40 pod, after data is fetched as in p1_train_one.ensure_cifar100):
    python scripts/collapse_on_activations.py
    python scripts/collapse_on_activations.py --batch-size 256 --ckpt path/to/sd.pt
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import platform
from pathlib import Path

import torch

from qkla.engine import set_seed
from qkla.hf_vit import build_hf_vit, count_params
from qkla.feature_maps import QuantumKernelFeatureMap

# tuned bandwidths picked by the equal-budget search -- IDENTICAL table to
# scripts/collapse_diag.py so the two panels are directly comparable.
TUNED = {1: 2.0, 2: 1.5, 3: 1.0, 4: 1.0, 6: 0.75, 8: 0.75}
FIXED_BW = 1.0          # qk_norm auto default; the fixed-bandwidth reference


def _autocast(precision, device):
    if precision == "bf16":
        return torch.autocast(device_type=device, dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type=device, dtype=torch.float16)
    return contextlib.nullcontext()


def eff_rank(gram: torch.Tensor) -> float:
    """Entropy / participation effective rank of a PSD Gram matrix.

    Same definition as collapse_diag.stats: eff_rank = exp(H(p)) with
    p = eig / sum(eig). Computed in float64 for stability.
    """
    g = gram.double()
    g = 0.5 * (g + g.t())                                   # symmetrise (fp drift)
    ev = torch.linalg.eigvalsh(g).clamp_min(1e-12)
    p = ev / ev.sum()
    return torch.exp(-(p * p.log()).sum()).item()


def _attach_phi_hooks(model):
    """Register forward hooks on every attention feature_map to capture phi(q).

    Returns (captured_q, captured_k, handles). The LinearViTSelfAttention calls
    feature_map(q, is_query=True) then feature_map(k, is_query=False), so within
    one model.forward each feature_map module fires twice; we route by the
    is_query kwarg recorded on the input.
    """
    captured_q: list[torch.Tensor] = []
    captured_k: list[torch.Tensor] = []

    def hook(module, args, kwargs, output):
        is_query = kwargs.get("is_query", False)
        if (not is_query) and len(args) >= 2:               # positional fallback
            is_query = bool(args[1])
        (captured_q if is_query else captured_k).append(output.detach())

    handles = []
    for layer in model.vit.encoder.layer:
        attn = layer.attention.attention
        fmap = getattr(attn, "feature_map", None)
        if isinstance(fmap, QuantumKernelFeatureMap):
            handles.append(fmap.register_forward_hook(hook, with_kwargs=True))
    if not handles:
        raise RuntimeError("no QuantumKernelFeatureMap hooks attached -- "
                           "is the variant 'quantum'?")
    return captured_q, captured_k, handles


def _pool_gram(feats: list[torch.Tensor], max_tokens: int) -> torch.Tensor:
    """Stack per-layer phi tensors (each (b,h,n,r)), flatten (b,h) into the token
    axis, subsample to <= max_tokens rows for a tractable n x n eigensolve, and
    return the real feature-map Gram phi phi^T (n x n) in float64 on CPU.

    This is the REAL analogue of collapse_diag's exact kernel matrix K: there it
    was the closed-form Gram of synthetic x; here it is phi(q) phi(q)^T of the
    model's actual q activations on a real image batch.
    """
    # feats[i]: (b, h, n, r) -> (b*h*n, r); concat layers along the token axis.
    rows = torch.cat([f.reshape(-1, f.shape[-1]) for f in feats], dim=0)
    rows = rows.float().cpu()
    n = rows.shape[0]
    if n > max_tokens:
        g = torch.Generator().manual_seed(0)
        sel = torch.randperm(n, generator=g)[:max_tokens]
        rows = rows[sel]
    return (rows.double() @ rows.double().t())              # (m, m) Gram


def _make_loader(dev, batch_size, dataset):
    """One real MedMNIST batch source, reusing the project's MedMNIST loader
    (GPU-resident on CUDA, DataLoader fallback on CPU). Matched to the RunPod
    MedMNIST-only experiment; CIFAR is the locked result and is not re-run here.
    """
    from scripts.medmnist_data import build_loaders
    tr, _ = build_loaders(dataset, dev, batch_size, workers=0)
    return tr


def _one_batch(loader, dev):
    for xb, yb in loader:
        return xb.to(dev, non_blocking=True)
    raise RuntimeError("empty MedMNIST loader")


def gram_for_depth(L, bw, xb, dev, precision, max_tokens, ckpt, base_cfg):
    """Build a depth-L quantum ViT at bandwidth bw, hook phi(q), forward xb once,
    return (eff_rank(phi(q)phi(q)^T), eff_rank(phi(k)phi(k)^T), n_tokens)."""
    set_seed(0)                                             # reproducible build
    model = build_hf_vit("quantum", layers=L, bandwidth=bw, **base_cfg).to(dev)
    if ckpt:
        sd = torch.load(ckpt, map_location=dev)
        sd = sd.get("model", sd.get("state_dict", sd))
        # CRITICAL: drop the feature-map buffers (_w/_b) from the checkpoint. They
        # encode the TRAINING (tuned) bandwidth; loading them would overwrite the
        # bandwidth-specific _w we just built, so the FIXED-bandwidth pass would
        # silently run at the trained bandwidth and the tuned-vs-fixed collapse
        # comparison would vanish. We want the TRAINED q/k/v Linears but the
        # bandwidth `bw` requested here -> load everything EXCEPT _w/_b.
        n_before = len(sd)
        sd = {k: v for k, v in sd.items()
              if not (k.endswith("._w") or k.endswith("._b"))}
        loaded = model.load_state_dict(sd, strict=False)
        # sanity: the q/k/v Linears must actually load, else activations are untrained
        n_lin = sum(1 for k in sd if k.endswith(("query.weight", "key.weight", "value.weight")))
        print(f"[E5] L{L} bw={bw}: loaded {len(sd)}/{n_before} tensors "
              f"({n_lin} q/k/v Linears), missing={len(loaded.missing_keys)}", flush=True)
        if n_lin == 0:                          # #8: fail LOUD, never measure untrained silently
            raise RuntimeError(
                f"ckpt {ckpt}: 0 q/k/v Linear weights present -> activations would be "
                "UNTRAINED. Aborting (bad or architecture-mismatched checkpoint).")
    model.eval()
    cap_q, cap_k, handles = _attach_phi_hooks(model)
    try:
        with torch.no_grad(), _autocast(precision, dev):
            model(pixel_values=xb)
    finally:
        for h in handles:
            h.remove()
    gq = _pool_gram(cap_q, max_tokens)
    gk = _pool_gram(cap_k, max_tokens)
    return eff_rank(gq), eff_rank(gk), gq.shape[0]


def main():
    p = argparse.ArgumentParser(description="E5: collapse diagnostic on real "
                                            "quantum-ViT phi(q) activations.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=512,
                   help="subsample pooled tokens to keep the n x n eigensolve cheap")
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--ckpt", default="",
                   help="single state_dict .pt (overrides --ckpt-dir for every depth)")
    p.add_argument("--ckpt-dir", default="",
                   help="dir of p1 --save-weights checkpoints; per depth L loads "
                        "<dataset>_quantum_L{L}_s{seed}.pt (TRAINED activations)")
    p.add_argument("--seed", type=int, default=0,
                   help="which trained seed's checkpoint to load from --ckpt-dir")
    p.add_argument("--dataset", default="pathmnist",
                   help="MedMNIST flag the trained models used (pathmnist/dermamnist)")
    p.add_argument("--out-dir", default="./results/p1")
    p.add_argument("--tag", default="collapse_on_activations")
    args = p.parse_args()

    def _ckpt_for(L):
        """Resolve the trained-weights checkpoint for quantum depth L (or None)."""
        if args.ckpt:
            return args.ckpt
        if not args.ckpt_dir:
            return None
        pfx = "" if args.dataset == "cifar100" else f"{args.dataset}_"
        cp = Path(args.ckpt_dir) / f"{pfx}quantum_L{L}_s{args.seed}.pt"
        return str(cp) if cp.exists() else None

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU"
    print(f"[E5] device={dev} ({gpu}) precision={args.precision}", flush=True)

    # the matched-budget ViT config p1 trains (num_classes from the dataset -- 9
    # for PathMNIST; the 6-layer backbone `depth` is the ViT block count. Here the
    # per-depth `layers` is the quantum qubit-depth knob, INDEPENDENT of the ViT
    # block depth -- do NOT confuse the two). Must match p1's build_hf_vit defaults
    # (hidden 192/depth 6/heads 3/mlp 384/r 256/dc 1.0/qk_norm) or the state_dict
    # Linears won't load; the per-depth load-count print guards that.
    from scripts.medmnist_data import num_classes as mm_classes
    base_cfg = dict(image_size=32, patch_size=4, num_classes=mm_classes(args.dataset),
                    hidden_size=192, depth=6, heads=3, mlp_dim=384,
                    num_features=256, dc=1.0, qk_norm=True)

    set_seed(0)
    loader = _make_loader(dev, args.batch_size, args.dataset)
    xb = _one_batch(loader, dev)
    print(f"[E5] real {args.dataset} batch x: {tuple(xb.shape)}", flush=True)

    _n_trained = sum(_ckpt_for(L) is not None for L in TUNED)
    print(f"[E5] trained checkpoints found: {_n_trained}/{len(TUNED)} depths "
          f"({'TRAINED activations' if _n_trained else 'UNTRAINED init -- pass --ckpt-dir'})",
          flush=True)

    print("\n=== COLLAPSE ON ACTIVATIONS (E5) ===", flush=True)
    print("H: per-depth tuned bandwidth equalizes the REAL feature-map Gram "
          "phi(q)phi(q)^T across depth -> eff-rank spread collapses.\n", flush=True)

    # [A] real-activation eff-rank at TUNED bandwidth (expect MATCHED across L).
    print("[A] real phi(q)phi(q)^T eff-rank at TUNED bw  (expect ~MATCHED)", flush=True)
    print(f"  {'L':>3} {'bw':>5} {'effrank_q':>10} {'effrank_k':>10} {'n_tok':>6}",
          flush=True)
    tuned_rows, tuned_er = [], []
    for L, bw in TUNED.items():
        erq, erk, ntok = gram_for_depth(L, bw, xb, dev, args.precision,
                                        args.max_tokens, _ckpt_for(L), base_cfg)
        tuned_er.append(erq)
        tuned_rows.append({"L": L, "bw": bw, "eff_rank_q": erq,
                           "eff_rank_k": erk, "n_tokens": ntok})
        print(f"  {L:>3} {bw:>5} {erq:>10.1f} {erk:>10.1f} {ntok:>6}", flush=True)
    tuned_spread = max(tuned_er) - min(tuned_er)
    print(f"  -> eff_rank_q spread across L (tuned): {tuned_spread:.1f}", flush=True)

    # [B] real-activation eff-rank at FIXED bw (expect DIVERGENCE = U-curve driver).
    print(f"\n[B] real phi(q)phi(q)^T eff-rank at FIXED bw={FIXED_BW}  "
          "(expect DIVERGENCE)", flush=True)
    print(f"  {'L':>3} {'bw':>5} {'effrank_q':>10} {'effrank_k':>10} {'n_tok':>6}",
          flush=True)
    fixed_rows, fixed_er = [], []
    for L in TUNED:
        erq, erk, ntok = gram_for_depth(L, FIXED_BW, xb, dev, args.precision,
                                        args.max_tokens, _ckpt_for(L), base_cfg)
        fixed_er.append(erq)
        fixed_rows.append({"L": L, "bw": FIXED_BW, "eff_rank_q": erq,
                           "eff_rank_k": erk, "n_tokens": ntok})
        print(f"  {L:>3} {FIXED_BW:>5} {erq:>10.1f} {erk:>10.1f} {ntok:>6}", flush=True)
    fixed_spread = max(fixed_er) - min(fixed_er)
    print(f"  -> eff_rank_q spread across L (fixed): {fixed_spread:.1f}", flush=True)

    # #7 margin: require fixed spread to CLEARLY exceed tuned (a bare `<` would call
    # 5.1-vs-5.0 a "collapse"). CIFAR's collapse was ~3x (234->76); we set the bar
    # at 1.5x so only a substantial collapse counts as CONFIRMED.
    MARGIN = 1.5
    ratio = fixed_spread / tuned_spread if tuned_spread > 1e-9 else float("inf")
    if ratio >= MARGIN:
        verdict = "CONFIRMED"
    elif fixed_spread > tuned_spread:
        verdict = f"WEAK (positive but < {MARGIN:g}x; not a clear collapse)"
    else:
        verdict = "NOT confirmed"
    confirmed = ratio >= MARGIN
    print(f"\n[VERDICT] collapse needs fixed-spread >= {MARGIN:g}x tuned-spread on REAL",
          flush=True)
    print(f"  activations: tuned={tuned_spread:.1f} vs fixed={fixed_spread:.1f} "
          f"(ratio={ratio:.2f}x) -> {verdict}", flush=True)

    # params (parity sanity -- the matched-budget invariant; layers/bw are free)
    ref_model = build_hf_vit("quantum", layers=1, bandwidth=FIXED_BW, **base_cfg)
    params = count_params(ref_model)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.tag}.json"
    payload = {
        "tag": args.tag,
        "diagnostic": "E5_collapse_on_activations",
        "description": "eff-rank of real phi(q)phi(q)^T per quantum depth L, "
                       "tuned vs fixed bandwidth; confirmation panel for the "
                       "collapse figure (real-activation analogue of "
                       "scripts/collapse_diag.py).",
        "variant": "quantum",
        "qk_norm": True,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "precision": args.precision,
        "ckpt_dir": args.ckpt_dir or None,
        "single_ckpt": args.ckpt or None,
        "trained_depths": _n_trained,
        "activations": "trained" if _n_trained else "untrained_init",
        "params": params,
        "fixed_bw": FIXED_BW,
        "tuned_bandwidths": TUNED,
        "tuned": tuned_rows,
        "fixed": fixed_rows,
        "tuned_spread": tuned_spread,
        "fixed_spread": fixed_spread,
        "spread_ratio": ratio,
        "collapse_margin": MARGIN,
        "verdict": verdict,
        "collapse_confirmed": confirmed,
        "env": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": gpu,
            "host": platform.node(),
            "python": platform.python_version(),
            "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[E5] DONE -> {path}", flush=True)


if __name__ == "__main__":
    main()

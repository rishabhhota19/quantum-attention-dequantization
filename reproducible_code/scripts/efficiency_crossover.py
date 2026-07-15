"""Efficiency crossover: softmax O(n^2) vs linear-feature-map O(n) attention.
Measures fwd+bwd latency (ms) and peak memory (MB) for the attention OP at growing
token counts n, at matched head dims. Softmax should blow up / OOM in the high-token
regime while the feature-map variant stays flat. Produces the Table tab:eff numbers.

Run: PYTHONPATH=. python scripts/efficiency_crossover.py
"""
import time, json, argparse
import torch
from qkla.feature_maps import QuantumKernelFeatureMap

p = argparse.ArgumentParser()
p.add_argument("--batch", type=int, default=4)
p.add_argument("--heads", type=int, default=3)
p.add_argument("--dim-head", type=int, default=64)
p.add_argument("--features", type=int, default=256)
p.add_argument("--iters", type=int, default=10)
p.add_argument("--ns", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
p.add_argument("--out", default="results/efficiency_crossover.json")
args = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
B, H, Dh, R = args.batch, args.heads, args.dim_head, args.features
gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
print(f"device={dev} ({gpu}) B={B} H={H} d={Dh} r={R}")

fm = QuantumKernelFeatureMap(Dh, R, bandwidth=1.0, layers=4).to(dev)

def softmax_attn(q, k, v):
    s = (q @ k.transpose(-1, -2)) / (Dh ** 0.5)
    return s.softmax(-1) @ v

def linear_attn(q, k, v):
    pq = fm(q, is_query=True); pk = fm(k)                       # (B,H,n,R)
    kv = torch.einsum("bhnr,bhnd->bhrd", pk, v)
    num = torch.einsum("bhnr,bhrd->bhnd", pq, kv)
    z = pk.sum(dim=2)
    den = torch.einsum("bhnr,bhr->bhn", pq, z).clamp_min(1e-6)
    return num / den.unsqueeze(-1)

def bench(fn, n):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    q = torch.randn(B, H, n, Dh, device=dev, requires_grad=True)
    k = torch.randn(B, H, n, Dh, device=dev, requires_grad=True)
    v = torch.randn(B, H, n, Dh, device=dev, requires_grad=True)
    try:
        for _ in range(3):                                      # warmup
            fn(q, k, v).sum().backward()
        if dev == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.iters):
            q.grad = k.grad = v.grad = None
            fn(q, k, v).sum().backward()
        if dev == "cuda": torch.cuda.synchronize()
        dt = (time.time() - t0) / args.iters * 1000
        mem = torch.cuda.max_memory_allocated() / 1e6 if dev == "cuda" else float("nan")
        return round(dt, 2), round(mem, 1)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache(); return "OOM", "OOM"
        raise

rows = {}
for n in args.ns:
    sm = bench(softmax_attn, n)
    ln = bench(linear_attn, n)
    rows[n] = {"softmax_ms": sm[0], "softmax_mb": sm[1], "linear_ms": ln[0], "linear_mb": ln[1]}
    print(f"n={n:5d}  softmax {str(sm[0]):>7} ms / {str(sm[1]):>8} MB   |   linear {str(ln[0]):>7} ms / {str(ln[1]):>8} MB")

import os; os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump({"device": gpu, "config": vars(args), "rows": rows}, open(args.out, "w"), indent=2)
print("wrote", args.out)

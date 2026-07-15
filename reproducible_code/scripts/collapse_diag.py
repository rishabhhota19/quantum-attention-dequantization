"""Collapse diagnostic — WHY do the shift-invariant variants collapse to ~44
once each is tuned its own bandwidth?

Hypothesis: per-variant bandwidth tuning lets every shift-invariant map land on
the SAME effective kernel smoothness -> identical inductive bias -> identical
accuracy. The depth U-curve only exists at a FIXED bandwidth; tuning erases it.

Evidence this prints:
1. tuned bandwidth vs depth L  -> monotone decrease (deeper compensates smaller bw)
2. exact kernel statistics (mean off-diagonal similarity + effective/entropy rank
   of the Gram) at the TUNED bw  -> should be ~matched across L (collapse), vs
   at a FIXED bw -> should diverge with L (the U-curve driver).

No training, no GPU needed; uses the closed-form exact quantum kernel.
"""
import torch
from qkla.feature_maps import quantum_kernel_matrix

# tuned bandwidths picked by the equal-budget search (best.json + uends)
TUNED = {1: 2.0, 2: 1.5, 3: 1.0, 4: 1.0, 6: 0.75, 8: 0.75}

DIM_HEAD = 64
N = 256
torch.manual_seed(0)
# unit-sphere inputs (post QK-norm regime the attention actually sees)
x = torch.randn(N, DIM_HEAD)
x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def stats(L, bw):
    K = quantum_kernel_matrix(x, bandwidth=bw, layers=L).double()
    off = K[~torch.eye(N, dtype=torch.bool)]
    ev = torch.linalg.eigvalsh(K).clamp_min(1e-12)
    p = ev / ev.sum()
    eff_rank = torch.exp(-(p * p.log()).sum()).item()   # entropy / participation rank
    return off.mean().item(), off.std().item(), eff_rank


print("\n=== COLLAPSE DIAGNOSTIC ===")
print("H: tuning bandwidth equalizes the effective kernel across depth -> accuracy collapses.\n")

print("[A] tuned bandwidth vs depth (compensation fingerprint)")
print(f"  {'L':>3} {'tuned_bw':>9}")
for L, bw in TUNED.items():
    print(f"  {L:>3} {bw:>9}")

print("\n[B] exact-kernel stats at TUNED bw  (expect ~MATCHED across L = collapse)")
print(f"  {'L':>3} {'bw':>5} {'mean_offK':>10} {'std_offK':>9} {'eff_rank':>9}")
tuned_er = []
for L, bw in TUNED.items():
    m, s, er = stats(L, bw)
    tuned_er.append(er)
    print(f"  {L:>3} {bw:>5} {m:>10.4f} {s:>9.4f} {er:>9.1f}")
print(f"  -> eff_rank spread across L (tuned): {max(tuned_er)-min(tuned_er):.1f}")

print("\n[C] exact-kernel stats at FIXED bw=1.0  (expect DIVERGENCE = the U-curve driver)")
print(f"  {'L':>3} {'bw':>5} {'mean_offK':>10} {'std_offK':>9} {'eff_rank':>9}")
fixed_er = []
for L in (1, 2, 3, 4, 6, 8):
    m, s, er = stats(L, 1.0)
    fixed_er.append(er)
    print(f"  {L:>3} {1.0:>5} {m:>10.4f} {s:>9.4f} {er:>9.1f}")
print(f"  -> eff_rank spread across L (fixed): {max(fixed_er)-min(fixed_er):.1f}")

print("\n[VERDICT] if tuned-spread << fixed-spread, the collapse is confirmed:")
print("  fair bandwidth tuning makes every depth approximate the same effective")
print("  kernel, so the 'depth helps' U-curve was a bandwidth-mismatch artifact.")

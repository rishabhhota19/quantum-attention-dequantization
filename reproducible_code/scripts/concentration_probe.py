"""IQP kernel concentration vs coupling c — direct measurement (no training).
As c grows, off-diagonal Gram std contracts (the exponential-concentration signature).
Run: PYTHONPATH=. python scripts/concentration_probe.py
"""
import torch
from qkla.feature_maps import IQPKernelFeatureMap
torch.manual_seed(0)
d, n, r = 64, 256, 8192
x = torch.randn(n, d); x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
print(f"{'c':>5} {'mean_offK':>10} {'std_offK':>9} {'eff_rank':>9}")
for c in [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]:
    fm = IQPKernelFeatureMap(d, r, bandwidth=1.0, coupling=c)
    phi = fm(x.view(1, 1, n, d)).view(n, r)
    K = (phi @ phi.t()).double()
    off = K[~torch.eye(n, dtype=torch.bool)]
    ev = torch.linalg.eigvalsh(K).clamp_min(1e-12); p = ev / ev.sum()
    er = torch.exp(-(p * p.log()).sum()).item()
    print(f"{c:>5} {off.mean().item():>10.4f} {off.std().item():>9.4f} {er:>9.1f}")

"""Generate the paper figures from the locked pod data (pod_data_2026-06-22/).
Figures we can build from data in hand:
  fig1_stage2_bar    : Stage 2 @70ep, all variants, mean +/- std (the tie)
  fig2_iqp_csweep    : IQP entanglement accuracy vs coupling c (tie + concentration drop)
  fig3_collapse      : effective-kernel rank vs depth L, tuned-bw vs fixed-bw (the U-curve is an artifact)
Run: python paper/make_paper_figures.py
"""
import json, glob, statistics as st
from collections import defaultdict
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "pod_data_2026-06-22/results"
OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)


def agg(folder):
    d = defaultdict(list)
    for f in glob.glob(f"{DATA}/{folder}/*.json"):
        if "summary" in f: continue
        d[os.path.basename(f).rsplit("_s", 1)[0]].append(json.load(open(f))["best_val_acc"])
    return d


# ---- fig1: Stage 2 bar ----
s2 = agg("stage2")
order = ["softmax","gaussian_rff","sign_rff","quantum_L1","quantum_L2","quantum_L3",
         "quantum_L4","quantum_L6","quantum_L8","performer"]
labels = ["softmax","gauss-RFF","sign-RFF","qL1","qL2","qL3","qL4","qL6","qL8","performer"]
means = [st.mean(s2[k]) for k in order]; errs = [st.pstdev(s2[k]) for k in order]
colors = ["#444"]+["#2a7"]*2+["#48c"]*6+["#c55"]
plt.figure(figsize=(8,4))
plt.bar(range(len(order)), means, yerr=errs, color=colors, capsize=3)
plt.axhline(st.mean(s2["gaussian_rff"]), ls="--", c="#2a7", lw=1, label="best generic RFF")
plt.xticks(range(len(order)), labels, rotation=40, ha="right")
plt.ylabel("CIFAR-100 top-1 (%)"); plt.ylim(46, 56)
plt.title("Stage 2 @70ep, 3 seeds — quantum ties generic RFF (softmax leads)")
plt.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/fig1_stage2_bar.png", dpi=150); plt.close()

# ---- fig2: IQP c-sweep ----
iqp = agg("iqp")
cs = ["0.0","0.1","0.3","0.5","1.0","2.0"]
ks = [f"iqp_c{c}" for c in cs]
m = [st.mean(iqp[k]) for k in ks]; e = [st.pstdev(iqp[k]) for k in ks]
plt.figure(figsize=(6.5,4))
plt.errorbar([float(c) for c in cs], m, yerr=e, marker="o", capsize=3, c="#48c")
plt.axhline(m[0], ls="--", c="#888", lw=1, label="no-entanglement control (c=0)")
plt.xlabel("entanglement coupling  c"); plt.ylabel("CIFAR-100 top-1 (%)")
plt.title("IQP entanglement: no separation, concentration hurts at high c")
plt.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/fig2_iqp_csweep.png", dpi=150); plt.close()

# ---- fig3: collapse (eff_rank vs L, tuned vs fixed) ----
# values from the collapse diagnostic (exact kernel; reproduced here for the figure)
import torch
from qkla.feature_maps import quantum_kernel_matrix
DH, N = 64, 256
torch.manual_seed(0)
x = torch.randn(N, DH); x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
def eff_rank(L, bw):
    K = quantum_kernel_matrix(x, bandwidth=bw, layers=L).double()
    ev = torch.linalg.eigvalsh(K).clamp_min(1e-12); p = ev/ev.sum()
    return torch.exp(-(p*p.log()).sum()).item()
Ls = [1,2,3,4,6,8]; tuned = {1:2.0,2:1.5,3:1.0,4:1.0,6:0.75,8:0.75}
er_tuned = [eff_rank(L, tuned[L]) for L in Ls]
er_fixed = [eff_rank(L, 1.0) for L in Ls]
plt.figure(figsize=(6.5,4))
plt.plot(Ls, er_fixed, "o-", c="#c55", label="fixed bandwidth=1.0 (the apparent U-curve)")
plt.plot(Ls, er_tuned, "s-", c="#2a7", label="per-depth tuned bandwidth (collapses)")
plt.xlabel("encoding depth  L"); plt.ylabel("effective rank of Gram kernel")
plt.title("Depth 'advantage' is a bandwidth artifact (eff-rank spread 234 -> 76)")
plt.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/fig3_collapse.png", dpi=150); plt.close()

print("wrote:", os.listdir(OUT))

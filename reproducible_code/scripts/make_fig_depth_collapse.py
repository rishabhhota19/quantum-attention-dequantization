"""Session-1 figure: the depth U-curve in ACCURACY space — untuned (fixed bandwidth)
vs per-depth tuned. Shows the inverted-U (peak L4) collapsing to flat under fair tuning.
Complements fig3 (effective-rank). Data: pod_data_2026-06-22 ucurve (untuned 10ep) + p1 (tuned 10ep).
"""
import json, glob, os, statistics as st
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "pod_data_2026-06-22/results"
def agg(folder):
    d = defaultdict(list)
    for f in glob.glob(f"{D}/{folder}/*.json"):
        if "summary" in f: continue
        d[os.path.basename(f).rsplit("_s", 1)[0]].append(json.load(open(f))["best_val_acc"])
    return d

u, p = agg("ucurve"), agg("p1")
Ls = [1, 2, 3, 4, 6, 8]
unt_m = [st.mean(u[f"qL{L}"]) for L in Ls];  unt_e = [st.pstdev(u[f"qL{L}"]) for L in Ls]
tun_m = [st.mean(p[f"quantum_L{L}"]) for L in Ls]; tun_e = [st.pstdev(p[f"quantum_L{L}"]) for L in Ls]
gen = st.mean(u["sign_rff"])  # best generic at untuned

plt.figure(figsize=(6.5, 4))
plt.errorbar(Ls, unt_m, yerr=unt_e, marker="o", capsize=3, c="#c55",
             label="untuned (fixed bandwidth) — inverted-U")
plt.errorbar(Ls, tun_m, yerr=tun_e, marker="s", capsize=3, c="#2a7",
             label="per-depth tuned — collapses to flat")
plt.axhline(gen, ls="--", c="#888", lw=1, label="best generic RFF")
plt.xlabel("encoding depth  L"); plt.ylabel("CIFAR-100 top-1 (%)  @10ep, 3 seeds")
plt.title("Depth 'advantage' is a tuning artifact: the U-curve flattens")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig("paper/figures/fig5_depth_untuned_vs_tuned.png", dpi=150); plt.close()
print("wrote paper/figures/fig5_depth_untuned_vs_tuned.png")
print("untuned spread:", round(max(unt_m) - min(unt_m), 2), " tuned spread:", round(max(tun_m) - min(tun_m), 2))

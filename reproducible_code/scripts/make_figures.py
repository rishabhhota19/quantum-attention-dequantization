"""Regenerate every figure as a saved PNG (paper evidence) from the result JSONs.

Prefers the P1 seeded run (results/p1/summary.json -> mean±std error bars) when
present; otherwise falls back to the 1-seed local screening
(results/layers_sweep.json, results/decision_10ep.json). Run anytime:

    python scripts/make_figures.py        # -> figures/*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path("figures"); FIG.mkdir(exist_ok=True)
RES = Path("results")
GREEN, BLUE, GRAY, RED = "#3B6D11", "#185FA5", "#888888", "#C0392B"
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})

# entanglement coupling sweep (2 ep, 1 seed) — measured, embedded for reproducibility
COUPLING_2EP = {0.0: 13.47, 0.05: 13.27, 0.1: 13.30, 0.25: 12.44, 0.5: 12.79, 1.0: 8.49}


def _load(path):
    p = RES / path
    return json.loads(p.read_text()) if p.exists() else None


def depth_data():
    """Return {L: (mean, std)} preferring P1, else 1-seed layers_sweep."""
    p1 = _load("p1/summary.json")
    if p1 and any(k.startswith("quantum_L") for k in p1):
        d = {}
        for k, v in p1.items():
            if k.startswith("quantum_L"):
                d[int(k.split("L")[1])] = (v["mean"], v["std"])
        return d, True
    ls = _load("layers_sweep.json") or {}
    d = {int(k.split("L")[1]): (v[0], 0.0) for k, v in ls.items() if k.startswith("quantum_L")}
    return d, False


def baselines():
    """generic + softmax reference levels (mean) from P1 or decision_10ep."""
    p1 = _load("p1/summary.json")
    if p1:
        g = max((p1[k]["mean"] for k in ("gaussian_rff", "sign_rff") if k in p1), default=None)
        return g, p1.get("softmax", {}).get("mean")
    d = _load("decision_10ep.json") or {}
    g = max((d[k][0] for k in ("gaussian_rff", "sign_rff(iqp_c0)") if k in d), default=None)
    return g, (d.get("softmax") or [None])[0]


def fig_depth():
    d, seeded = depth_data()
    if not d:
        return
    gen, sm = baselines()
    Ls = sorted(d); m = [d[L][0] for L in Ls]; e = [d[L][1] for L in Ls]
    fig, ax = plt.subplots(figsize=(7, 4.3))
    if gen:
        ax.axhspan(gen, max(m) + 3, color=GREEN, alpha=0.06)
        ax.axhline(gen, ls="--", color="#1D9E75", lw=1.4, label=f"best generic {gen:.1f}")
    if sm:
        ax.axhline(sm, ls=":", color=GRAY, lw=1.4, label=f"softmax {sm:.1f}")
    ax.errorbar(Ls, m, yerr=e if seeded else None, marker="o", color=BLUE, lw=2,
                capsize=4, label="quantum (prod)")
    peak = max(Ls, key=lambda L: d[L][0])
    ax.plot(peak, d[peak][0], "*", ms=18, color=GREEN, zorder=5)
    ax.annotate(f"peak L{peak} = {d[peak][0]:.1f}", (peak, d[peak][0]),
                textcoords="offset points", xytext=(0, 12), ha="center", color=GREEN)
    ax.set_xlabel("qubit depth L  (data re-uploading)"); ax.set_ylabel("CIFAR-100 top-1 %")
    ax.set_title(f"Depth sweet spot — {'P1 (mean±std)' if seeded else '1-seed screen'}")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.15)
    fig.tight_layout(); fig.savefig(FIG / "fig1_depth_sweet_spot.png"); plt.close(fig)


def fig_depth_vs_entanglement():
    d, _ = depth_data(); gen, _ = baselines()
    if not d:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
    Ls = sorted(d)
    a1.axhline(gen or 34.26, ls="--", color=GRAY, lw=1.2)
    a1.plot(Ls, [d[L][0] for L in Ls], "o-", color=GREEN, lw=2)
    a1.set_title("depth — helps"); a1.set_xlabel("L"); a1.set_ylabel("top-1 %"); a1.grid(alpha=0.15)
    ent = _load("ent/summary.json")               # prefer seeded P-ENT if present
    if ent:
        cs = sorted(float(k) for k in ent)
        cm = [ent[str(c)]["mean"] for c in cs]; ce = [ent[str(c)]["std"] for c in cs]
        a2.errorbar(cs, cm, yerr=ce, fmt="o-", color=RED, lw=2, capsize=4)
        a2.axhline(cm[0], ls="--", color=GRAY, lw=1.2, label="c=0 control")
    else:
        cs = sorted(COUPLING_2EP)
        a2.axhline(COUPLING_2EP[0.0], ls="--", color=GRAY, lw=1.2, label="c=0 control")
        a2.plot(cs, [COUPLING_2EP[c] for c in cs], "o-", color=RED, lw=2)
    a2.set_title("entanglement — hurts"); a2.set_xlabel("coupling c"); a2.grid(alpha=0.15)
    a2.legend(frameon=False, fontsize=9)
    fig.suptitle("Two knobs, two fates: depth (Fourier richness) vs entanglement (concentration)")
    fig.tight_layout(); fig.savefig(FIG / "fig2_depth_vs_entanglement.png"); plt.close(fig)


def fig_main_bar():
    p1 = _load("p1/summary.json")
    if p1:
        labels = [k for k in ("softmax", "performer", "gaussian_rff", "sign_rff",
                              "quantum_L2", "quantum_L3", "quantum_L4") if k in p1]
        m = [p1[k]["mean"] for k in labels]; e = [p1[k]["std"] for k in labels]
    else:
        d = _load("decision_10ep.json") or {}
        order = ["softmax", "gaussian_rff", "sign_rff(iqp_c0)", "quantum(prod_L1)",
                 "quantum(prod_L2)", "quantum(prod_L3)" if "quantum(prod_L3)" in d else None]
        labels = [k for k in order if k and k in d]; m = [d[k][0] for k in labels]; e = None
    if not labels:
        return
    colors = [GREEN if "quantum" in l else GRAY for l in labels]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(range(len(labels)), m, yerr=e, color=colors, alpha=0.85, capsize=4)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("CIFAR-100 top-1 %"); ax.set_title("Matched-budget comparison (1.82M params)")
    ax.grid(alpha=0.15, axis="y"); fig.tight_layout()
    fig.savefig(FIG / "fig3_main_bar.png"); plt.close(fig)


def fig_training_curves():
    files = sorted((RES / "p1").glob("*_s0.json")) if (RES / "p1").exists() else []
    if not files:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for f in files:
        d = json.loads(f.read_text())
        if d.get("history"):
            ax.plot([h["epoch"] for h in d["history"]], [h["val_acc"] for h in d["history"]],
                    lw=1.6, label=d["tag"].replace("_s0", ""))
    ax.set_xlabel("epoch"); ax.set_ylabel("val top-1 %"); ax.set_title("Training curves (seed 0)")
    ax.legend(frameon=False, fontsize=8, ncol=2); ax.grid(alpha=0.15)
    fig.tight_layout(); fig.savefig(FIG / "fig4_training_curves.png"); plt.close(fig)


if __name__ == "__main__":
    for fn in (fig_depth, fig_depth_vs_entanglement, fig_main_bar, fig_training_curves):
        try:
            fn()
        except Exception as e:
            print(f"{fn.__name__}: skipped ({e})")
    print("figures ->", ", ".join(p.name for p in sorted(FIG.glob("*.png"))) or "(none)")

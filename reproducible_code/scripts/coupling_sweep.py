"""P0.1 — the real entanglement test: does ANY coupling beat c=0?

The 6-variant ablation showed full-strength entanglement (c=1) HURTS (~5 pts),
consistent with quantum-kernel concentration (strong ZZ phase -> kernel
concentrates). That does NOT settle whether *weak* entanglement helps. This
sweeps coupling c in {0, .05, .1, .25, .5, 1} for the iqp kernel (everything
else fixed, one knob) and reports the c-vs-accuracy curve.

  c=0            : no entanglement (sign-RFF control)
  best c > c=0   : weak entanglement HELPS  -> win path
  c=0 is best    : entanglement only hurts  -> honest-pivot path

gaussian_rff is included once as the generic-RFF reference line.
"""

from __future__ import annotations

from scripts.iqp_ablation import run

COUPLINGS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]


def main(epochs=2):
    specs = [("gaussian_rff", {"variant": "gaussian_rff"})]
    specs += [(f"iqp_c={c}", {"variant": "iqp", "coupling": c}) for c in COUPLINGS]
    rows = run(specs, epochs=epochs)

    acc = {c: rows[f"iqp_c={c}"][0] for c in COUPLINGS}
    c0 = acc[0.0]
    best_c = max(COUPLINGS, key=lambda c: acc[c])
    print(f"\n=== coupling sweep ({epochs} ep) ===", flush=True)
    for c in COUPLINGS:
        mark = "  <- best" if c == best_c else ""
        print(f"  c={c:<5} acc={acc[c]:5.2f}  (vs c=0: {acc[c]-c0:+.2f}){mark}", flush=True)
    print(f"  gaussian_rff ref = {rows['gaussian_rff'][0]:.2f}", flush=True)
    if best_c != 0.0 and acc[best_c] > c0 + 0.3:           # small margin guard
        print(f"VERDICT: weak entanglement HELPS — best c={best_c} beats c=0 "
              f"by {acc[best_c]-c0:+.2f} (-> win path, converge it on L40)", flush=True)
    else:
        print(f"VERDICT: entanglement does NOT help here — c=0 best; "
              f"entanglement only hurts (-> honest-pivot path / concentration story)", flush=True)


if __name__ == "__main__":
    main()

"""Improve the product quantum kernel by sweeping qubit depth L.

L is a FREE, FAIR knob: it only reshapes the angle-kernel's frequency-sampling
distribution (per-coord accessible frequencies {-L..L}), so params / FLOPs / r
are unchanged. Theory: a quantum model is a Fourier series in the data and
repeating the encoding L times linearly extends the accessible spectrum ->
more expressive (Schuld-Sweke-Meyer, arXiv:2008.08605). Empirically L1->L2 gave
+1.43 @10ep (32.64 -> 34.07, tied with generic).

Caveat: richer != always better (spectrum flattens, high-freqs vanish;
arXiv:2311.10822, 2510.14217) -> expect a SWEET SPOT in L, not monotone. This
sweep finds it. If the peak clears the generic bar (~34.2), it is the win path.

Checkpointed to results/layers_sweep.json (resumable).
"""

from __future__ import annotations

from scripts.iqp_ablation import run

LAYERS = [1, 2, 3, 4, 6, 8]


def main(epochs=10):
    specs = [("gaussian_rff", {"variant": "gaussian_rff"})]
    specs += [(f"quantum_L{L}", {"variant": "quantum", "layers": L}) for L in LAYERS]
    rows = run(specs, epochs=epochs, results_path="results/layers_sweep.json")

    acc = {L: rows[f"quantum_L{L}"][0] for L in LAYERS}
    gen = rows["gaussian_rff"][0]
    best = max(LAYERS, key=lambda L: acc[L])
    print(f"\n=== qubit-depth (L) sweep, {epochs} ep ===", flush=True)
    for L in LAYERS:
        mark = "  <- best" if L == best else ""
        print(f"  L={L}: {acc[L]:5.2f}  (vs generic {acc[L]-gen:+.2f}){mark}", flush=True)
    print(f"  gaussian_rff (generic) = {gen:.2f}", flush=True)
    if acc[best] > gen + 0.3:
        print(f"VERDICT: quantum(prod L={best}) = {acc[best]:.2f} BEATS generic by "
              f"{acc[best]-gen:+.2f} -> WIN PATH (converge + seeds + bandwidth tune on L40)",
              flush=True)
    else:
        print(f"VERDICT: best quantum L={best} ({acc[best]:.2f}) does NOT clear generic "
              f"({gen:.2f}) -> quantum structure competitive but not winning on accuracy",
              flush=True)


if __name__ == "__main__":
    main()

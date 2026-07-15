"""P0.5 — fair, equal-budget hyperparameter search (kills the baseline-bias hole).

The #1 reviewer objection: "you tuned your knob (L) but ran the generic RFFs
untuned." Fix: EVERY model gets a search of the SAME size (6 trials, 1 seed,
short epochs) over ITS natural hyperparameter, then the seeded P1 final runs each
model at its own tuned best. So the comparison is best-vs-best — no bias.

  softmax / performer  -> 6-point learning-rate search
  gaussian_rff / sign_rff (c=0) / quantum_L2 / quantum_L3 / quantum_L4
                       -> 6-point bandwidth search  (each L tuned independently)

Writes results/tune/best.json = {label: {hp: value}}, consumed by run_p1.py.

    python scripts/run_tune.py --tune-epochs 25 --parallel 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.run_p1 import run_pool   # shared isolated-process pool

BW6 = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
LR6 = [3e-4, 5e-4, 7e-4, 1e-3, 1.5e-3, 2e-3]

# label -> (search_key, [values], fixed kwargs). Equal budget: 6 trials each.
SEARCH = {
    "softmax":      ("lr",        LR6, {"variant": "softmax"}),
    "performer":    ("lr",        LR6, {"variant": "performer"}),
    "gaussian_rff": ("bandwidth", BW6, {"variant": "gaussian_rff"}),
    "sign_rff":     ("bandwidth", BW6, {"variant": "iqp", "coupling": 0.0}),
    "quantum_L2":   ("bandwidth", BW6, {"variant": "quantum", "layers": 2}),
    "quantum_L3":   ("bandwidth", BW6, {"variant": "quantum", "layers": 3}),
    "quantum_L4":   ("bandwidth", BW6, {"variant": "quantum", "layers": 4}),
}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--tune-epochs", type=int, default=25)
    p.add_argument("--parallel", type=int, default=6)
    p.add_argument("--precision", default="bf16")
    p.add_argument("--out-dir", default="./results/tune")
    p.add_argument("--log-dir", default="./logs/tune")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for label, (key, vals, kw) in SEARCH.items():
        for v in vals:
            tag = f"{label}__{key}{v}"
            f = out / f"{tag}.json"
            if f.exists() and json.loads(f.read_text()).get("done"):
                continue
            cmd = [sys.executable, "scripts/p1_train_one.py", "--tag", tag, "--seed", "0",
                   "--epochs", str(args.tune_epochs), "--precision", args.precision,
                   "--out-dir", args.out_dir, f"--{key}", str(v)]
            for k, vv in kw.items():
                cmd += [f"--{k.replace('_','-')}", str(vv)]
            jobs.append((tag, cmd))

    print(f"[tune] {len(jobs)} trials ({args.parallel} parallel), {args.tune_epochs} ep, equal 6-trial budget/model", flush=True)
    if args.dry_run:
        for t, c in jobs:
            print("  ", " ".join(c))
        return
    run_pool(jobs, args.parallel, args.log_dir)

    # pick best HP per model
    best = {}
    for label, (key, vals, kw) in SEARCH.items():
        trials = []
        for v in vals:
            f = out / f"{label}__{key}{v}.json"
            if f.exists() and json.loads(f.read_text()).get("done"):
                trials.append((v, json.loads(f.read_text())["best_val_acc"]))
        if trials:
            bv, bacc = max(trials, key=lambda t: t[1])
            best[label] = {key: bv, "tune_acc": round(bacc, 2),
                           "searched": {str(v): round(a, 2) for v, a in trials}}
    (out / "best.json").write_text(json.dumps(best, indent=2))
    print("\n[tune] best configs (equal-budget search):")
    for label, b in best.items():
        print(f"  {label:14s} {b}", flush=True)
    print(f"-> {out/'best.json'}  (run_p1.py consumes this)")


if __name__ == "__main__":
    main()

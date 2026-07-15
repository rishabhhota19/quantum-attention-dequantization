"""P1 orchestrator — the confirmation run that decides win vs pivot.

Grid: {softmax, performer, gaussian_rff, sign_rff(iqp c=0), quantum L2/L3/L4}
x SEEDS, 100 epochs, bf16, matched budget. Runs each (variant,seed) as an
ISOLATED subprocess in a parallel pool of `--parallel` workers so the L40 is
filled WITHOUT harming training (independent processes -> identical math, only
GPU time is shared; accuracy is unaffected). Each job checkpoints itself and is
skipped on restart. After the grid, aggregates mean+/-std, writes the P1 table,
and regenerates all figures.

NOTE: timing/efficiency (M4) must be measured ISOLATED (--parallel 1) for valid
wall-clock; P1 is an accuracy run, so parallelism is safe here.

    python scripts/run_p1.py --epochs 100 --seeds 0 1 2 3 4 --parallel 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev

VARIANTS = [
    ("softmax",      {"variant": "softmax"}),
    ("performer",    {"variant": "performer"}),
    ("gaussian_rff", {"variant": "gaussian_rff"}),
    ("sign_rff",     {"variant": "iqp", "coupling": 0.0}),     # entanglement-off control
    ("quantum_L2",   {"variant": "quantum", "layers": 2}),
    ("quantum_L3",   {"variant": "quantum", "layers": 3}),     # the peak to confirm
    ("quantum_L4",   {"variant": "quantum", "layers": 4}),
]
GENERIC = ["gaussian_rff", "sign_rff"]
QUANTUM = ["quantum_L2", "quantum_L3", "quantum_L4"]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--precision", default="bf16")
    p.add_argument("--parallel", type=int, default=4, help="concurrent jobs (L40: 4-8)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers (use 0 on Windows)")
    p.add_argument("--out-dir", default="./results/p1")
    p.add_argument("--log-dir", default="./logs/p1")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _tuned():
    """Load equal-budget tuned HPs (results/tune/best.json) if present -> {label:{hp}}."""
    f = Path("results/tune/best.json")
    if not f.exists():
        return {}
    out = {}
    for label, b in json.loads(f.read_text()).items():
        out[label] = {k: v for k, v in b.items() if k in ("lr", "bandwidth")}
    return out


def build_jobs(args):
    tuned = _tuned()
    if tuned:
        print(f"[P1] using equal-budget tuned HPs for {list(tuned)} (no baseline bias)", flush=True)
    jobs = []
    for label, kw in VARIANTS:
        kw = {**kw, **tuned.get(label, {})}                 # tuned HP overrides defaults
        for seed in args.seeds:
            tag = f"{label}_s{seed}"
            done = Path(args.out_dir) / f"{tag}.json"
            if done.exists() and json.loads(done.read_text()).get("done"):
                continue
            cmd = [sys.executable, "scripts/p1_train_one.py", "--tag", tag,
                   "--seed", str(seed), "--epochs", str(args.epochs),
                   "--precision", args.precision, "--batch-size", str(args.batch_size),
                   "--out-dir", args.out_dir]
            for k, v in kw.items():
                cmd += [f"--{k.replace('_','-')}", str(v)]
            jobs.append((tag, cmd))
    return jobs


def run_pool(jobs, parallel, log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    running, i = [], 0
    while i < len(jobs) or running:
        while i < len(jobs) and len(running) < parallel:
            tag, cmd = jobs[i]; i += 1
            log = open(Path(log_dir) / f"{tag}.log", "w")
            print(f"[pool] start {tag}  ({len(running)+1}/{parallel} slots)", flush=True)
            running.append((tag, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        time.sleep(5)
        for tag, proc, log in running[:]:
            if proc.poll() is not None:
                log.close(); running.remove((tag, proc, log))
                print(f"[pool] done  {tag}  (exit {proc.returncode})", flush=True)


def aggregate(args):
    out = Path(args.out_dir)
    agg = {}
    for label, _ in VARIANTS:
        accs = []
        for seed in args.seeds:
            f = out / f"{label}_s{seed}.json"
            if f.exists() and json.loads(f.read_text()).get("done"):
                accs.append(json.loads(f.read_text())["best_val_acc"])
        if accs:
            agg[label] = {"mean": mean(accs), "std": stdev(accs) if len(accs) > 1 else 0.0,
                          "n": len(accs), "accs": accs}
    (out / "summary.json").write_text(json.dumps(agg, indent=2))

    lines = [f"# P1 — confirmation run (CIFAR-100, bf16, {args.epochs} ep, matched budget)",
             f"seeds={args.seeds}", "", "| variant | top-1 % (mean±std) | n |",
             "|---|---|---|"]
    for label, _ in VARIANTS:
        if label in agg:
            a = agg[label]
            lines.append(f"| {label} | {a['mean']:.2f} ± {a['std']:.2f} | {a['n']} |")
    bq = max((l for l in QUANTUM if l in agg), key=lambda l: agg[l]["mean"], default=None)
    bg = max((l for l in GENERIC if l in agg), key=lambda l: agg[l]["mean"], default=None)
    if bq and bg:
        d = agg[bq]["mean"] - agg[bg]["mean"]
        pooled = (agg[bq]["std"] ** 2 + agg[bg]["std"] ** 2) ** 0.5
        verdict = "WIN (>2σ)" if d > 2 * pooled else ("edge (<2σ)" if d > 0 else "no win")
        lines += ["", f"**make-or-break:** best quantum ({bq} {agg[bq]['mean']:.2f}) − "
                  f"best generic ({bg} {agg[bg]['mean']:.2f}) = {d:+.2f}  "
                  f"(±{pooled:.2f} pooled) → **{verdict}**"]
    text = "\n".join(lines)
    (out / "P1_table.md").write_text(text, encoding="utf-8")
    try:
        print(text)
    except UnicodeEncodeError:                          # Windows cp1252 console
        print(text.encode("ascii", "replace").decode())


def main():
    args = parse()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    print(f"[P1] {len(jobs)} jobs to run ({args.parallel} parallel), bf16, {args.epochs} ep", flush=True)
    if args.dry_run:
        for tag, cmd in jobs:
            print("  ", " ".join(cmd))
        return
    run_pool(jobs, args.parallel, args.log_dir)
    aggregate(args)
    try:
        subprocess.run([sys.executable, "scripts/make_figures.py"], check=False)
    except Exception as e:
        print(f"[P1] figure gen skipped: {e}")


if __name__ == "__main__":
    main()

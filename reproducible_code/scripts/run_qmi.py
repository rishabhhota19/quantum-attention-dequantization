"""QMI P0 orchestrator — the cross-dataset confirmation run (CIFAR-100 + MedMNIST).

QMI ("quantum vs matched-budget, two datasets") is the P0 experiment: re-run the
locked variant subset on BOTH datasets at the SAME 100-epoch budget so the
quantum-vs-generic comparison is fair by construction (matched params, bf16,
identical schedule; the ONLY thing that moves is the feature-map knob). Fairness
rule #10: we RE-RUN every variant at 100ep — we NEVER reuse the old 70ep
checkpoints as a baseline for the new quantum runs.

Variant subset (locked):
    softmax     — trusted HF baseline, lr 5e-4
    gaussian_rff— classical RBF cos-RFF control, bandwidth 1.5      (generic)
    sign_rff    — IQP with coupling=0 (entanglement OFF), bandwidth 1.0 (generic)
    quantum     — angle-embedding product kernel, layers=4, bandwidth 1.0 (quantum)
    iqp_c0.3    — IQP with coupling=0.3 (entanglement ON), bandwidth 1.0 (quantum)
x seeds {0,1,2,3,4} x dataset {pathmnist} @ 100 epochs (MedMNIST-only; CIFAR is locked, not re-run).

Each (variant, seed, dataset) runs as an ISOLATED subprocess (independent
processes -> identical math, only GPU time is shared; accuracy unaffected) by
shelling out to scripts/p1_train_one.py with the matching flags + a --dataset
flag (assumes p1_train_one.py gains --dataset). Every job checkpoints itself
(resumable: a complete json is skipped on restart). After the grid, aggregates a
per-dataset mean+/-std table and prints the make-or-break verdict
(best quantum - best generic) for EACH dataset.

This driver can fan out the jobs locally in a Popen pool (--parallel), or just
emit a ready-to-paste pod launch command that fans out with `xargs -P 4`
(--print-launch). On the A40 use PAR=4 (PAR>=12 thrashes the A40); orchestrate
with xargs -P4, NOT bash "jobs -r" (which fails under nohup).

POD DEPS (install on the A40 before launching):
    pip install --break-system-packages \
        transformers==4.57.6 einops performer_pytorch pyarrow pillow \
        huggingface_hub medmnist
    (transformers v5 renames ViTSelfAttention -> pin 4.57.6; container disk
     wipes on restart so re-install every boot.)

Usage:
    # emit the pod xargs -P4 launch command (no training; safe anywhere):
    python scripts/run_qmi.py --print-launch
    # fan out locally / on the pod via a Python Popen pool:
    python scripts/run_qmi.py --epochs 100 --seeds 0 1 2 3 4 \
        --datasets pathmnist --parallel 4   # MedMNIST-only; CIFAR is the locked result, not re-run
    # aggregate only (after the xargs grid finished):
    python scripts/run_qmi.py --aggregate-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev

# ---------------------------------------------------------------------------
# Locked variant subset. Each entry: (label, {p1_train_one flags}). The flags
# are passed verbatim to scripts/p1_train_one.py (key -> --key, '_' -> '-').
# Feature maps are parameter-free, so every variant has exactly the softmax
# trainable-param count -- EXCEPT qpsan (E4), whose faithful QPSAN variational
# angles add dim_head params per layer (~384 total, <0.03% of 1.8M): a documented
# ~matched budget, not an unfair capacity advantage.
# ---------------------------------------------------------------------------
VARIANTS = [
    ("softmax",      {"variant": "softmax", "lr": 5e-4}),
    ("gaussian_rff", {"variant": "gaussian_rff", "bandwidth": 1.5}),
    ("sign_rff",     {"variant": "iqp", "coupling": 0.0, "bandwidth": 1.0}),  # entanglement OFF
    ("quantum_L4",   {"variant": "quantum", "layers": 4, "bandwidth": 1.0}),
    ("iqp_c0.3",     {"variant": "iqp", "coupling": 0.3, "bandwidth": 1.0}),  # entanglement ON
    # E4 (qpsan) DETACHED from the runnable list 2026-06-30 (budget) -- the
    # QPSANFeatureMap code + hf_vit "qpsan" variant remain intact; re-add this
    # line to run it. The identical 5-variant list (above) is the locked retrace.
    # ("qpsan",        {"variant": "qpsan", "bandwidth": 1.0}),  # E4: parameterised external control
]

# Buckets for the make-or-break verdict (best quantum - best generic).
GENERIC = ["gaussian_rff", "sign_rff"]
QUANTUM = ["quantum_L4", "iqp_c0.3"]

# Concrete dataset flags passed verbatim to p1_train_one's --dataset (whose
# choices are cifar100/pathmnist/dermamnist). NB: "medmnist" is NOT a valid
# p1 flag -- the second dataset must be a real MedMNIST task. PathMNIST is the
# locked second benchmark (9-class colon pathology); swap to "dermamnist" here
# for the HAM10000 task instead.
# MedMNIST-ONLY: CIFAR-100 is the locked, already-completed result (70ep/3seed) and
# is NOT re-run here. This experiment is a faithful reproduction of the IDENTICAL rig
# on a second dataset only — same variants, same tuned HPs, same matched budget, same
# bf16; only the data distribution changes. PathMNIST is the locked second benchmark.
DATASETS = ["pathmnist"]
SEEDS = [0, 1, 2, 3, 4]


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--datasets", nargs="+", default=DATASETS,
                   choices=["cifar100", "pathmnist", "dermamnist"])
    p.add_argument("--precision", default="bf16",
                   help="bf16 only — fp16 collapses Performer/IQP (#3)")
    p.add_argument("--parallel", type=int, default=4,
                   help="concurrent jobs for the local Popen pool (A40: PAR=4; >=12 thrashes, #8)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=0,
                   help="DataLoader workers; 0 on Windows / GPU-resident path (#7)")
    p.add_argument("--out-dir", default="./results/qmi")
    p.add_argument("--log-dir", default="./logs/qmi")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--print-launch", action="store_true",
                   help="print the ready-to-paste pod 'xargs -P4' launch command and exit")
    p.add_argument("--aggregate-only", action="store_true",
                   help="skip training, just aggregate existing checkpoints into the tables")
    p.add_argument("--dry-run", action="store_true",
                   help="print the per-job commands the pool would run, then exit")
    p.add_argument("--stop-pod", action="store_true",
                   help="power the RunPod pod off when the grid + aggregate finish")
    return p.parse_args()


def tag_for(label: str, dataset: str, seed: int) -> str:
    """Stable checkpoint tag. os.path.basename-safe; no slashes (#1)."""
    return f"{dataset}_{label}_s{seed}"


def job_command(args, label: str, kw: dict, dataset: str, seed: int):
    """The exact argv to train ONE (variant, dataset, seed). Mirrors run_p1.py."""
    tag = tag_for(label, dataset, seed)
    cmd = [sys.executable, "scripts/p1_train_one.py",
           "--tag", tag,
           "--dataset", dataset,
           "--seed", str(seed),
           "--epochs", str(args.epochs),
           "--precision", args.precision,
           "--batch-size", str(args.batch_size),
           "--workers", str(args.workers),
           "--data-dir", args.data_dir,
           "--out-dir", args.out_dir]
    for k, v in kw.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    return tag, cmd


def is_done(out_dir: str, tag: str) -> bool:
    f = Path(out_dir) / f"{tag}.json"
    if not f.exists():
        return False
    try:
        return bool(json.loads(f.read_text(encoding="utf-8")).get("done"))
    except (json.JSONDecodeError, OSError):
        return False


def build_jobs(args):
    """All (variant, dataset, seed) jobs not yet complete -> [(tag, cmd), ...]."""
    jobs = []
    for dataset in args.datasets:
        for label, kw in VARIANTS:
            for seed in args.seeds:
                tag = tag_for(label, dataset, seed)
                if is_done(args.out_dir, tag):
                    continue
                jobs.append(job_command(args, label, kw, dataset, seed))
    return jobs


# ---------------------------------------------------------------------------
# Pod launch command (xargs -P4). Emits a single shell line per job into a job
# file, then one `xargs -P 4` invocation that fans them out. We deliberately do
# NOT use bash "jobs -r" (it fails under nohup, #8).
# ---------------------------------------------------------------------------
def print_launch(args):
    # Each job line carries its own per-job log redirect, so plain `xargs -P 4`
    # gives per-(variant,seed,dataset) logs without any %-substitution gymnastics.
    lines = []
    for dataset in args.datasets:
        for label, kw in VARIANTS:
            for seed in args.seeds:
                tag, cmd = job_command(args, label, kw, dataset, seed)
                # cmd[0] is sys.executable on THIS host; on the pod use python3.
                shell = " ".join(["python3"] + cmd[1:])
                lines.append(f"{shell} > {args.log_dir}/{tag}.log 2>&1")

    jobs_file = "qmi_jobs.sh"
    n = len(lines)
    print("# ---------------------------------------------------------------------------")
    print("# QMI P0 -- ready-to-paste A40 pod launch (xargs -P 4)")
    print(f"# {n} jobs = {len(VARIANTS)} variants x {len(args.seeds)} seeds "
          f"x {len(args.datasets)} datasets @ {args.epochs} ep, bf16, matched budget")
    print("# Run from the reproducible_code/ root. Deps (re-install each boot;")
    print("# container disk wipes on restart):")
    print("#   pip install --break-system-packages transformers==4.57.6 einops \\")
    print("#       performer_pytorch pyarrow pillow huggingface_hub medmnist")
    print("# ---------------------------------------------------------------------------")
    print(f"mkdir -p {args.out_dir} {args.log_dir}")
    print(f"cat > {jobs_file} <<'EOF'")
    for s in lines:
        print(s)
    print("EOF")
    print()
    print("# PAR=4 fills the A40 without thrashing (#8). Each line redirects to its own")
    print("# log; -I CMD feeds exactly one full job line to each bash -c (NOT 'jobs -r', #8).")
    print("# Resumable: completed (variant,seed,dataset) jsons are auto-skipped (#9).")
    print(f"nohup xargs -P 4 -I CMD bash -c CMD < {jobs_file} "
          f"> {args.log_dir}/qmi_all.log 2>&1 &")
    print()
    print("# When the grid is done, aggregate the per-dataset tables + verdicts:")
    print(f"#   python3 scripts/run_qmi.py --aggregate-only "
          f"--datasets {' '.join(args.datasets)} --seeds {' '.join(map(str, args.seeds))} "
          f"--out-dir {args.out_dir}")


# ---------------------------------------------------------------------------
# Local Popen pool (mirrors run_p1.run_pool: parallel isolated subprocesses).
# ---------------------------------------------------------------------------
def run_pool(jobs, parallel, log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    running, i = [], 0
    while i < len(jobs) or running:
        while i < len(jobs) and len(running) < parallel:
            tag, cmd = jobs[i]; i += 1
            log = open(Path(log_dir) / f"{tag}.log", "w", encoding="utf-8")  # #4
            print(f"[pool] start {tag}  ({len(running) + 1}/{parallel} slots)", flush=True)
            running.append((tag, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        time.sleep(5)
        for tag, proc, log in running[:]:
            if proc.poll() is not None:
                log.close(); running.remove((tag, proc, log))
                print(f"[pool] done  {tag}  (exit {proc.returncode})", flush=True)


# ---------------------------------------------------------------------------
# Aggregation: per-dataset mean+/-std table + make-or-break verdict.
# ---------------------------------------------------------------------------
def aggregate_dataset(out_dir: str, dataset: str, seeds, epochs):
    out = Path(out_dir)
    agg = {}
    for label, _ in VARIANTS:
        accs = []
        for seed in seeds:
            f = out / f"{tag_for(label, dataset, seed)}.json"
            if f.exists():
                d = json.loads(f.read_text(encoding="utf-8"))
                if d.get("done"):
                    accs.append(d["best_val_acc"])
        if accs:
            agg[label] = {"mean": mean(accs),
                          "std": stdev(accs) if len(accs) > 1 else 0.0,
                          "n": len(accs), "accs": accs}
    (out / f"summary_{dataset}.json").write_text(json.dumps(agg, indent=2),
                                                 encoding="utf-8")

    lines = [f"## QMI P0 — {dataset} (bf16, {epochs} ep, matched budget, "
             f"seeds={list(seeds)})",
             "",
             "| variant | bucket | top-1 % (mean+/-std) | n |",
             "|---|---|---|---|"]
    bucket = {**{l: "generic" for l in GENERIC},
              **{l: "QUANTUM" for l in QUANTUM},
              "softmax": "reference"}
    for label, _ in VARIANTS:
        if label in agg:
            a = agg[label]
            lines.append(f"| {label} | {bucket.get(label, '')} | "
                         f"{a['mean']:.2f} +/- {a['std']:.2f} | {a['n']} |")

    bq = max((l for l in QUANTUM if l in agg), key=lambda l: agg[l]["mean"], default=None)
    bg = max((l for l in GENERIC if l in agg), key=lambda l: agg[l]["mean"], default=None)
    verdict_line = None
    if bq and bg:
        d = agg[bq]["mean"] - agg[bg]["mean"]
        pooled = (agg[bq]["std"] ** 2 + agg[bg]["std"] ** 2) ** 0.5
        min_n = min(agg[bq]["n"], agg[bg]["n"])
        if min_n < 2 or pooled == 0.0:
            # single-seed (or zero-variance) -> NO significance test; 2*pooled=0
            # would otherwise declare a spurious ">2 sigma WIN" on any +delta.
            verdict = f"inconclusive (need >=2 seeds for significance; n={min_n})"
        elif d > 2 * pooled:
            verdict = "WIN (>2 sigma)"
        elif d > 0:
            verdict = "edge (<2 sigma)"
        else:
            verdict = "no win"
        verdict_line = (f"**make-or-break [{dataset}]:** best quantum ({bq} "
                        f"{agg[bq]['mean']:.2f}) - best generic ({bg} "
                        f"{agg[bg]['mean']:.2f}) = {d:+.2f} "
                        f"(+/-{pooled:.2f} pooled) -> **{verdict}**")
        lines += ["", verdict_line]
    return agg, "\n".join(lines), verdict_line


def aggregate(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    blocks, verdicts = [], []
    header = [f"# QMI P0 — quantum vs matched-budget generic, two datasets",
              f"epochs={args.epochs}  seeds={list(args.seeds)}  precision={args.precision}  "
              f"matched params (assert_param_parity)", ""]
    for dataset in args.datasets:
        _, block, verdict = aggregate_dataset(args.out_dir, dataset, args.seeds, args.epochs)
        blocks.append(block)
        if verdict:
            verdicts.append(verdict)
    text = "\n".join(header + blocks + ([""] if verdicts else []) +
                     (["### Verdict summary", ""] + verdicts if verdicts else []))
    (out / "QMI_table.md").write_text(text, encoding="utf-8")
    try:
        print(text)
    except UnicodeEncodeError:                              # Windows cp1252 console (#4)
        print(text.encode("ascii", "replace").decode())
    print(f"\n-> {out / 'QMI_table.md'}")


def main():
    args = parse()

    if args.print_launch:
        print_launch(args)
        return

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        aggregate(args)
        return

    jobs = build_jobs(args)
    n_total = len(VARIANTS) * len(args.seeds) * len(args.datasets)
    print(f"[QMI] {len(jobs)}/{n_total} jobs to run "
          f"({args.parallel} parallel), bf16, {args.epochs} ep, "
          f"datasets={args.datasets}", flush=True)
    if args.dry_run:
        for tag, cmd in jobs:
            print("  ", " ".join(cmd))
        return

    run_pool(jobs, args.parallel, args.log_dir)
    aggregate(args)

    if args.stop_pod:
        print("[QMI] run complete -> powering pod off", flush=True)
        subprocess.run([sys.executable, "scripts/stop_pod.py"], check=False)


if __name__ == "__main__":
    main()

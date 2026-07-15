#!/usr/bin/env bash
# E5 -- collapse-on-trained-activations, POD run (CIFAR-matched, single seed).
# 6 quantum DEPTHS {1,2,3,4,6,8}, 1 seed, 100 ep, --save-weights (weights kept!),
# then the collapse diagnostic, then bundle + auto-stop. Needs RUNPOD_API_KEY in
# the env for the auto-stop (export it at launch; NOT hardcoded so it never lands
# in the repo). Run from /workspace/exp. 6 jobs, ALL in one wave (PAR=6 + NVIDIA
# MPS, bit-identical speedup) -> ~4-5h A40 instead of ~8h, ~$2-3.
set -u
cd /workspace/exp
export PYTHONPATH=. PYTHONUTF8=1
mkdir -p results/e5 logs/e5

echo "[e5] $(date -u +%F_%H:%M) START depth sweep L{1,2,3,4,6,8} x1 seed x100ep +save-weights"

# pre-fetch PathMNIST ONCE (serially) so the 6 PAR=6 jobs don't race on a cold
# download to the same ./data/medmnist files. Idempotent; a no-op if cached.
python -c "from scripts.medmnist_data import ensure_medmnist; ensure_medmnist('pathmnist'); print('[e5] pathmnist ready')" \
  || { echo "[e5] FATAL: pathmnist prefetch failed"; python scripts/stop_pod.py; exit 1; }
# tuned bandwidth per depth (identical to the collapse diagnostic's TUNED map)
cat > e5_jobs.sh <<'JOBS'
python scripts/p1_train_one.py --variant quantum --layers 1 --bandwidth 2.0  --dataset pathmnist --epochs 100 --seed 0 --save-weights --out-dir results/e5 > logs/e5/L1.log 2>&1
python scripts/p1_train_one.py --variant quantum --layers 2 --bandwidth 1.5  --dataset pathmnist --epochs 100 --seed 0 --save-weights --out-dir results/e5 > logs/e5/L2.log 2>&1
python scripts/p1_train_one.py --variant quantum --layers 3 --bandwidth 1.0  --dataset pathmnist --epochs 100 --seed 0 --save-weights --out-dir results/e5 > logs/e5/L3.log 2>&1
python scripts/p1_train_one.py --variant quantum --layers 4 --bandwidth 1.0  --dataset pathmnist --epochs 100 --seed 0 --save-weights --out-dir results/e5 > logs/e5/L4.log 2>&1
python scripts/p1_train_one.py --variant quantum --layers 6 --bandwidth 0.75 --dataset pathmnist --epochs 100 --seed 0 --save-weights --out-dir results/e5 > logs/e5/L6.log 2>&1
python scripts/p1_train_one.py --variant quantum --layers 8 --bandwidth 0.75 --dataset pathmnist --epochs 100 --seed 0 --save-weights --out-dir results/e5 > logs/e5/L8.log 2>&1
JOBS

# --- SPEEDUP (bit-identical: no experiment/kernel change) ---------------------
# The 6 jobs are tiny (1.8M params, seq 65) and individually starve the A40, so we
# (a) run ALL 6 in ONE wave (PAR=6) and (b) put them under NVIDIA MPS for true
# concurrent (spatial) GPU sharing instead of time-slicing. Each job's computation
# is untouched -> weights and the collapse are identical; only wall-time drops.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
MPS=0
if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  if nvidia-cuda-mps-control -d 2>/dev/null; then MPS=1; echo "[e5] NVIDIA MPS daemon started (concurrent GPU sharing)"; fi
fi
[ "$MPS" = 0 ] && echo "[e5] MPS unavailable -> PAR=6 without it (still one wave, still identical)"

xargs -P6 -I CMD bash -c CMD < e5_jobs.sh    # all 6 depths concurrently, one wave

[ "$MPS" = 1 ] && { echo quit | nvidia-cuda-mps-control 2>/dev/null; echo "[e5] MPS daemon stopped"; }

echo "[e5] $(date -u +%F_%H:%M) training done"
# #10: surface any depth whose training crashed (its .pt would be absent -> the
# collapse would fall back to UNTRAINED for that depth). Loud warning, no abort.
n_pt=$(ls results/e5/pathmnist_quantum_L*_s0.pt 2>/dev/null | wc -l)
echo "[e5] trained weight files: $n_pt/6"
[ "$n_pt" -lt 6 ] && echo "[e5] WARNING: only $n_pt/6 depths trained -- check logs/e5/L*.log; collapse will use UNTRAINED activations for the missing depth(s)."

echo "[e5] $(date -u +%F_%H:%M) -> collapse diagnostic"
python scripts/collapse_on_activations.py --ckpt-dir results/e5 --dataset pathmnist --seed 0 --out-dir results/e5

echo "[e5] $(date -u +%F_%H:%M) collapse done -> bundle + auto-stop"
tar czf "e5results_$(date +%Y%m%d_%H%M).tar.gz" results/e5 logs/e5 2>/dev/null
python scripts/stop_pod.py            # resolves the live RUNNING pod via REST; needs RUNPOD_API_KEY in env
echo "[e5] $(date -u +%F_%H:%M) E5 DONE"

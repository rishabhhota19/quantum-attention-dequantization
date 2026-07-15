# Reproducible code

Everything needed to reproduce the study: the feature-map family, the matched-budget ViT rig, the
training/tuning pipeline, and the diagnostics.

## Setup
```bash
pip install -r requirements.txt
```
CIFAR-100 is downloaded automatically from the HuggingFace CDN on first run (parquet). A CUDA GPU is
recommended; the whole dataset is held resident in VRAM for high utilisation.

## The `qkla` package
| file | what it provides |
|---|---|
| `feature_maps.py` | the φ family — `ShiftInvariantRFF` base (Bochner cos features), `GaussianRFFBochner` (RBF control), `QuantumKernelFeatureMap` (angle/product, depth `L`), `IQPKernelFeatureMap` (entangling, coupling `c`) — plus the exact-kernel validators (`quantum_kernel_matrix/fidelity`, `iqp_kernel_matrix/fidelity`) |
| `hf_vit.py` | `build_hf_vit(variant, ...)` — a standard ViT whose attention operator is swapped per variant; all variants parameter-matched (1,823,908), reusing identical q/k/v projections |
| `gpu_data.py` | whole-dataset-in-VRAM CIFAR loader with on-GPU augmentation |
| `engine.py`, `models.py`, `data.py`, `linear_attention.py` | training utilities and supporting components |

## Scripts
| script | purpose |
|---|---|
| `run_tune.py` | equal-budget (6-point) hyperparameter search per variant → `results/tune/best.json` |
| `p1_train_one.py` | train one `(variant, seed)` to convergence; checkpoints per-epoch history + best accuracy |
| `run_p1.py` | orchestrate all variants × seeds → aggregate table with the make-or-break verdict |
| `layers_sweep.py` | depth `L` sweep (the inverted-U / collapse) |
| `coupling_sweep.py` | entanglement coupling `c` sweep |
| `collapse_diag.py` | exact-kernel effective-rank diagnostic (untuned-vs-tuned bandwidth) |
| `concentration_probe.py` | direct measurement of IQP kernel concentration vs `c` (off-diagonal std) |
| `efficiency_crossover.py` | O(n²) softmax vs O(n) linear attention: latency + peak memory vs token count |
| `make_paper_figures.py`, `make_fig_depth_collapse.py` | regenerate the figures from logged results |

## Reproduce the headline
```bash
# 1) fair tuning (short budget)
PYTHONPATH=. python scripts/run_tune.py
# 2) best-vs-best at the headline budget, seeded
PYTHONPATH=. python scripts/run_p1.py --epochs 70 --seeds 0 1 2
# 3) diagnostics (no training)
PYTHONPATH=. python scripts/collapse_diag.py
PYTHONPATH=. python scripts/concentration_probe.py
PYTHONPATH=. python scripts/efficiency_crossover.py
```
Note: the figure scripts expect a results directory; point them at `../evidence/results` or your own run
output.

## Validation
The feature maps reproduce their quantum kernels: closed form vs. exact statevector simulation agrees to
~1e-7 in the default fp32 quick check (float64 gives <1e-14 — the figure reported in the paper), and the
random-feature error decays at the Monte-Carlo 1/√r rate (verifiable via
`quantum_kernel_fidelity` / `iqp_kernel_fidelity`; `scripts/pennylane_crossval.py` is the independent
PennyLane cross-check).

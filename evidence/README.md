# Evidence

All logged results behind the study, plus the figures and a numbers summary. Every number in the paper
traces to a file here.

## Headline numbers (CIFAR-100 top-1 %, 70 epochs, 3 seeds)
- **softmax 54.92 ± 0.32** (leads)
- best quantum **qL4 52.55 ± 0.37**; best generic **gaussian_rff 52.36 ± 0.43** → **Δ = +0.19 (a tie)**
- IQP entanglement: best `c=0.3` 52.42 vs no-entanglement `c=0` 52.25 → **+0.17 (a tie)**; declines at
  high coupling (`c=1.0` 51.82, `c=2.0` 51.36)
- collapse diagnostic: effective-rank spread across depth **234 (fixed bw) → 76 (tuned)**
- concentration: IQP off-diagonal kernel std contracts **0.037 (c=0) → 0.013 (c=2)**, ~3×

Full breakdown in [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md).

## `results/` — per-run JSON (one file per variant × seed)
| folder | what it is |
|---|---|
| `stage2/` | the headline best-vs-best @70ep, 3 seeds (Table 1) |
| `iqp/` | the entanglement coupling sweep @70ep, 3 seeds (Table 2) |
| `ucurve/` | the **untuned** depth runs @10ep (the inverted-U) |
| `p1/` | the **tuned** depth runs @10ep (the flattened curve) |
| `tune/`, `iqp_tune/`, `uends/` | the equal-budget hyperparameter search grids |

Each JSON records the config, environment, per-epoch history, and `best_val_acc`. Every variant is
parameter-matched at 1,823,908.

## `figures/`
| file | shows |
|---|---|
| `fig1_stage2_bar.png` | the tie — every quantum depth clusters with the classical RFF baselines; softmax leads |
| `fig5_depth_untuned_vs_tuned.png` | the depth inverted-U (untuned) collapsing to flat (tuned), in accuracy |
| `fig3_collapse.png` | the effective-rank diagnostic behind the collapse |
| `fig2_iqp_csweep.png` | entanglement: no separation, decline at high coupling |

## Reproducing the figures
The figure scripts in `../reproducible_code/scripts/` read these JSONs (adjust the data path to
`evidence/results`).

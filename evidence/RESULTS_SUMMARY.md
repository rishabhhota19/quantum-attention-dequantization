# RESULTS SUMMARY — locked numbers for the paper

All numbers are from logged runs in `pod_data_2026-06-22/` (NVIDIA A40, bf16, CIFAR-100,
matched params **1,823,908** across every variant, num_features r=256, batch 256,
cosine+warmup). Two budgets: 10ep (tuning + Stage A) and 70ep (Stage 2 headline).
**Do not invent numbers — everything the paper claims must trace to a JSON here.**

## Setup (fixed across all variants — the fairness backbone)
- Backbone: HuggingFace ViT, hidden 192, 3 heads, dim_head 64, patch 4, 32px CIFAR-100.
- Only the attention OP differs between variants (softmax vs linear feature maps). Params identical.
- Every variant gets an **equal-budget HP search** (6-point): lr for softmax/performer, bandwidth for the RFF/quantum family. Tuned at 10ep, run long at 70ep ("tune short, train long").
- Quantum φ verified against statevector sim: closed-form == statevector to 4e-7; RFF error → 1/√r.

## Tuned hyperparameters (from results/tune + iqp_tune)
softmax lr=5e-4 · performer lr=1e-3 · gaussian_rff bw=1.5 · sign_rff(IQP c=0) bw=1.0
quantum L1/L2/L3/L4/L6/L8 bw = 2.0/1.5/1.0/1.0/0.75/0.75  (note: monotone ↓ with depth = the collapse fingerprint)
IQP c=0/0.1/0.3/0.5/1.0/2.0 bw = 1.0/1.0/1.0/1.0/1.0/0.75

## TABLE 1 — Stage 2 @70ep, 3 seeds (mean ± std)  [fig1]
| variant | top-1 % |
|---|---|
| softmax        | 54.92 ± 0.32 |
| quantum_L4     | 52.55 ± 0.37  ← best quantum |
| quantum_L6     | 52.53 ± 0.23 |
| quantum_L8     | 52.48 ± 0.22 |
| quantum_L1     | 52.44 ± 0.14 |
| quantum_L3     | 52.41 ± 0.55 |
| gaussian_rff   | 52.36 ± 0.43  ← best generic |
| sign_rff       | 52.25 ± 0.15 |
| quantum_L2     | 52.09 ± 0.14 |
| performer      | 48.65 ± 0.55 |

**Make-or-break #1 (depth):** best quantum − best generic = **+0.19** (≪1σ) → TIE.
Softmax leads the field by ~2.4. U-curve flat (all quantum L1–L8 within 52.1–52.6).
Cross-check: identical verdict at 10ep (+0.41) — convergence does not change it.

## TABLE 2 — IQP entanglement c-sweep @70ep, 3 seeds (mean ± std)  [fig2]
| coupling c | tuned bw | top-1 % |
|---|---|---|
| 0.0 (no ent) | 1.0 | 52.25 ± 0.15 |
| 0.1 | 1.0 | 52.07 ± 0.33 |
| 0.3 | 1.0 | 52.42 ± 0.23  ← best |
| 0.5 | 1.0 | 52.39 ± 0.20 |
| 1.0 | 1.0 | 51.82 ± 0.13 |
| 2.0 | 0.75 | 51.36 ± 0.32 |

**Make-or-break #2 (entanglement):** best c (0.3, 52.42) − no-ent control (c=0, 52.25) = **+0.17** (≪1σ) → TIE.
High coupling HURTS monotonically (c≥0.5): textbook exponential concentration.
Kernel fact: ‖K_iqp(c=1) − K_iqp(c=0)‖/‖K(c=0)‖ = 0.354 (the entangled kernel IS 35% different — it carries real cross-coordinate structure — yet that structure buys no accuracy).

## TABLE 3 — Collapse diagnostic (exact kernel, no training)  [fig3]
Effective (entropy) rank of the Gram across depth L:
| L | tuned-bw eff_rank | fixed bw=1.0 eff_rank |
|---|---|---|
| 1 | 152.6 | 12.3 |
| 2 | 174.2 | 45.8 |
| 3 | 97.7  | 97.7 |
| 4 | 150.8 | 150.8 |
| 6 | 118.1 | 220.3 |
| 8 | 173.2 | 246.2 |
| **spread** | **76.5** | **234.0** |

**Verdict:** at fixed bandwidth, effective rank diverges monotonically with depth (spread 234) — this is the
"depth helps" U-curve. Under per-depth bandwidth tuning the spread collapses to 76.5 (no monotone trend) →
**the depth advantage is a bandwidth-mismatch artifact**, exactly what Bochner predicts for a shift-invariant
(→ dequantizable) kernel.

## ONE-LINE VERDICT
Both quantum levers fail at matched budget: depth (Fourier richness) collapses under fair tuning; entanglement
(cross-coordinate structure) gives no separation and concentrates at high coupling. The angle/IQP quantum
feature maps **dequantize to classical random features** in ViT attention.

## STILL TO RUN before submission (scaffolding — see PAPER_FOUNDATION.md)
- [ ] #5 datasets: MedMNIST + ImageNet-100 (defends the negative against "only CIFAR-100")
- [ ] #2 efficiency crossover table (O(n) wall-clock + peak memory vs softmax, 2k–4k tokens)
- [ ] #3 concentration curve (quantify the c≥0.5 decline — kernel variance/gap vs c)  [partly in fig2/Table2]
- [ ] #4 (optional/ceiling) MRMF dot-product RFF baseline OR port one literature embedding into the rig

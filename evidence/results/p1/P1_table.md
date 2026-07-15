# P1 — confirmation run (CIFAR-100, bf16, 10 ep, matched budget)
seeds=[0, 1, 2]

| variant | top-1 % (mean±std) | n |
|---|---|---|
| softmax | 44.92 ± 0.54 | 3 |
| performer | 35.53 ± 0.46 | 3 |
| gaussian_rff | 44.08 ± 0.22 | 3 |
| sign_rff | 44.02 ± 0.83 | 3 |
| quantum_L2 | 44.12 ± 0.97 | 3 |
| quantum_L3 | 43.74 ± 0.40 | 3 |
| quantum_L4 | 44.49 ± 0.76 | 3 |

**make-or-break:** best quantum (quantum_L4 44.49) − best generic (gaussian_rff 44.08) = +0.42  (±0.79 pooled) → **edge (<2σ)**
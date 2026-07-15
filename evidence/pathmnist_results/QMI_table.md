# QMI P0 — quantum vs matched-budget generic, two datasets
epochs=100  seeds=[0, 1, 2, 3, 4]  precision=bf16  matched params (assert_param_parity)

## QMI P0 — pathmnist (bf16, 100 ep, matched budget, seeds=[0, 1, 2, 3, 4])

| variant | bucket | top-1 % (mean+/-std) | n |
|---|---|---|---|
| softmax | reference | 85.94 +/- 0.59 | 5 |
| gaussian_rff | generic | 84.57 +/- 1.00 | 5 |
| sign_rff | generic | 84.43 +/- 0.99 | 5 |
| quantum_L4 | QUANTUM | 85.13 +/- 0.66 | 5 |
| iqp_c0.3 | QUANTUM | 84.86 +/- 0.61 | 5 |

**make-or-break [pathmnist]:** best quantum (quantum_L4 85.13) - best generic (gaussian_rff 84.57) = +0.56 (+/-1.20 pooled) -> **edge (<2 sigma)**

### Verdict summary

**make-or-break [pathmnist]:** best quantum (quantum_L4 85.13) - best generic (gaussian_rff 84.57) = +0.56 (+/-1.20 pooled) -> **edge (<2 sigma)**
# Quantum-Inspired Feature Maps Dequantize in Vision Transformer Attention
### A Matched-Budget Study

Open-source release of the study: a fair, parameter-matched evaluation of quantum-kernel feature maps as
linear-attention operators in a standard Vision Transformer.

## TL;DR
We implement quantum-kernel feature maps — an **angle/product** map (encoding-depth knob `L`) and an
**entangling IQP** map (coupling knob `c`) — as drop-in linear-attention operators, holding parameters,
FLOPs, feature dimension, and tuning budget fixed across every variant. At matched budget,
**neither quantum lever beats tuned classical random features**:

- **Depth ties** classical RFF; its apparent "advantage" is a **bandwidth-tuning artifact** that collapses
  under per-depth tuning (effective-rank spread across depth 234 → 76 on the exact kernel).
- **Entanglement** yields no separation from a non-entangled control and **degrades at high coupling**,
  accompanied by a directly measured ~3× contraction of the kernel's off-diagonal spread (concentration).
- The **tie replicates on PathMNIST** (MedMNIST v2) at 100 epochs over 5 seeds (85.13±0.66 vs 84.57±1.00),
  and the bandwidth collapse reproduces on the **trained model's own activations** (9.6× spread ratio).
- A standard **softmax ViT leads** the linear family by ~2.4 points on CIFAR-100 (~0.8 on PathMNIST); the
  linear family's value is the O(n) efficiency crossover, not accuracy.

The angle/product kernel is shift-invariant and therefore, by Bochner's theorem, an exact random-feature
kernel — the structural reason it cannot escape the classical RFF family. We make **no claim of physical
quantum advantage**: the method is classical and the quantum kernel is only its derivation.

## Repository layout
| folder | contents |
|---|---|
| [`reproducible_code/`](reproducible_code/) | the `qkla` package (feature maps + validators) and scripts to tune, train, diagnose, and profile every variant |
| [`evidence/`](evidence/) | all logged results (per-run JSON), the figures, and a numbers summary |

See each folder's `README.md` for details.

> **Note:** the `qpsan` variant present in the code is an external control (a QKSAN/QPSAN-style scoring,
> experiment E4). It is functional but **not reported in the paper**.

## Citation
> Rishabh Hota. *Quantum-Inspired Feature Maps Dequantize in Vision Transformer Attention: A
> Matched-Budget Study.* 2026.

## License
MIT — see [LICENSE](LICENSE).

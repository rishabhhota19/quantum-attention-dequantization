"""QKLA — Quantum-Kernel Linear Attention.

A quantum-inspired, GPU-native linear-attention mechanism for Vision
Transformers. Classical method; the quantum kernel supplies the inductive bias
via the feature map's projection sampler. Never claims physical quantum
advantage; the quantum kernel is only the derivation.
"""

from .feature_maps import (
    FeatureMap, GaussianRFF, ShiftInvariantRFF, GaussianRFFBochner,
    QuantumKernelFeatureMap, QuantumKernelRFF, IQPKernelFeatureMap, FEATURE_MAPS,
    quantum_kernel_matrix, quantum_kernel_fidelity,
    iqp_kernel_matrix, iqp_kernel_fidelity,
)
from .linear_attention import LinearAttention

__all__ = [
    "FeatureMap", "GaussianRFF", "ShiftInvariantRFF", "GaussianRFFBochner",
    "QuantumKernelFeatureMap", "QuantumKernelRFF", "IQPKernelFeatureMap", "FEATURE_MAPS",
    "quantum_kernel_matrix", "quantum_kernel_fidelity",
    "iqp_kernel_matrix", "iqp_kernel_fidelity",
    "LinearAttention",
]

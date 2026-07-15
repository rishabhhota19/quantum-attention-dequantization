"""Model factory: a single ViT with a swappable attention operator.

Every variant is the SAME upstream `vit_pytorch.vit.ViT`; only the attention
submodule of each transformer block changes. The feature maps carry no trainable
parameters, so all variants have IDENTICAL parameter counts -- the fairness
invariant the whole paper rests on (protocol step 1).

Variants
--------
softmax        : upstream Attention, unmodified -- the reference baseline.
performer      : LinearAttention + GaussianRFF      (FAVOR+ positive features).
gaussian_rff   : LinearAttention + GaussianRFFBochner(RBF cos-RFF -- M1 control).
quantum        : LinearAttention + QuantumKernelFeatureMap (the quantum map under study).

The M1 make-or-break comparison is `quantum` vs `gaussian_rff`: same cos-feature
code path, identical r, one knob (the spectral sampler).
"""

from __future__ import annotations

import torch

from vit_pytorch.vit import ViT
from .linear_attention import LinearAttention
from .feature_maps import (
    GaussianRFF, GaussianRFFBochner, QuantumKernelFeatureMap,
)

ATTENTION_VARIANTS = ("softmax", "performer", "gaussian_rff", "quantum")


def _make_feature_map(variant, dim_head, num_features, bandwidth, layers, dc, qk_norm):
    if variant == "performer":
        return GaussianRFF(dim_head, num_features)          # already positive
    if variant == "gaussian_rff":
        return GaussianRFFBochner(dim_head, num_features, bandwidth=bandwidth, dc=dc,
                                  qk_norm=qk_norm)
    if variant == "quantum":
        return QuantumKernelFeatureMap(dim_head, num_features, bandwidth=bandwidth,
                                       layers=layers, dc=dc, qk_norm=qk_norm)
    raise ValueError(f"no feature map for variant {variant!r}")


def build_vit(variant: str, *, image_size=32, patch_size=4, num_classes=100,
              dim=192, depth=6, heads=3, dim_head=64, mlp_dim=384,
              num_features=256, bandwidth=None, layers=1, dc=1.0, qk_norm=True,
              dropout=0., emb_dropout=0.) -> ViT:
    """Build a ViT whose attention operator is `variant`. All variants share an
    identical parameter count by construction."""
    if variant not in ATTENTION_VARIANTS:
        raise ValueError(f"variant must be one of {ATTENTION_VARIANTS}, got {variant!r}")

    model = ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes,
                dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim,
                dropout=dropout, emb_dropout=emb_dropout)

    if variant != "softmax":
        for blk in model.transformer.layers:
            fmap = _make_feature_map(variant, dim_head, num_features, bandwidth, layers,
                                     dc, qk_norm)
            blk[0] = LinearAttention(dim, heads, dim_head, dropout=dropout,
                                     feature_map=fmap, num_features=num_features)
    return model


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def assert_param_parity(cfg: dict, variants=ATTENTION_VARIANTS) -> int:
    """Build every variant and assert identical trainable-param counts. Returns
    the shared count. Cheap fairness guard to run before any sweep."""
    counts = {v: count_params(build_vit(v, **cfg)) for v in variants}
    ref = counts["softmax"]
    bad = {v: c for v, c in counts.items() if c != ref}
    assert not bad, f"PARAM MISMATCH vs softmax={ref:,}: {bad}"
    return ref

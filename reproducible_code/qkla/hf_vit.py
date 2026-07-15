"""HuggingFace-backed ViT for fully-trusted baselines.

The softmax baseline is HuggingFace `transformers` ViT, UNMODIFIED -- so the
reference we claim to beat is a standard, third-party implementation, not ours
(a reviewer cannot say we sandbagged our own baseline). The linear variants
subclass HF's `ViTSelfAttention` and REUSE its exact query/key/value Linear
layers, swapping only the attention core op (softmax -> phi(q)(phi(k)^T v)).
Because the projections are reused verbatim, every variant has an identical
parameter count and identical initialisation -- only the operator differs.

The only code that remains ours is the quantum feature map (the contribution)
and the ~5-line linear reassociation -- both unavoidable.
"""

from __future__ import annotations

import torch
from torch import nn

from transformers import ViTConfig, ViTForImageClassification
from transformers.models.vit.modeling_vit import ViTSelfAttention

from .feature_maps import (
    GaussianRFF, GaussianRFFBochner, QuantumKernelFeatureMap, IQPKernelFeatureMap,
    QPSANFeatureMap,
)

# "qpsan" = E4 external quantum-attention control (QKSAN/QPSAN-style PQC scoring),
# a published-method baseline kept param-free on the same linear-attention path.
HF_VARIANTS = ("softmax", "performer", "gaussian_rff", "quantum", "iqp", "qpsan")


def _make_feature_map(variant, dim_head, num_features, bandwidth, layers, dc, qk_norm,
                      coupling):
    if variant == "performer":
        return GaussianRFF(dim_head, num_features)
    if variant == "gaussian_rff":
        return GaussianRFFBochner(dim_head, num_features, bandwidth=bandwidth, dc=dc,
                                  qk_norm=qk_norm)
    if variant == "quantum":
        return QuantumKernelFeatureMap(dim_head, num_features, bandwidth=bandwidth,
                                       layers=layers, dc=dc, qk_norm=qk_norm)
    if variant == "iqp":                                  # entangling (ZZ/IQP)
        return IQPKernelFeatureMap(dim_head, num_features, bandwidth=bandwidth,
                                   coupling=coupling, dc=dc, qk_norm=qk_norm)
    if variant == "qpsan":                                # E4 external control (QKSAN)
        # `layers` doubles as the data re-uploading depth R (param-free, no new CLI flag).
        return QPSANFeatureMap(dim_head, num_features, bandwidth=bandwidth,
                               reuploads=layers, dc=dc, qk_norm=qk_norm)
    raise ValueError(variant)


class LinearViTSelfAttention(ViTSelfAttention):
    """HF ViTSelfAttention with the softmax core replaced by linear attention.

    Inherits (and reuses) HF's query/key/value Linears verbatim, so params and
    init are identical to the untouched softmax module. Returns the context in
    HF's (batch, seq, heads, head_size) convention so the parent wrapper merges
    heads exactly as for softmax."""

    def __init__(self, config, feature_map, eps: float = 1e-6):
        super().__init__(config)
        self.feature_map = feature_map        # param-free
        self.eps = eps

    def forward(self, hidden_states, head_mask=None):
        b = hidden_states.shape[0]
        shape = b, -1, self.num_attention_heads, self.attention_head_size
        q = self.query(hidden_states).view(*shape).transpose(1, 2)   # (b,h,n,d)
        k = self.key(hidden_states).view(*shape).transpose(1, 2)
        v = self.value(hidden_states).view(*shape).transpose(1, 2)

        phi_q = self.feature_map(q, is_query=True)                   # (b,h,n,r)
        phi_k = self.feature_map(k, is_query=False)
        kv = torch.einsum('b h n r, b h n d -> b h r d', phi_k, v)
        num = torch.einsum('b h n r, b h r d -> b h n d', phi_q, kv)
        z = phi_k.sum(dim=2)                                         # (b,h,r)
        den = torch.einsum('b h n r, b h r -> b h n', phi_q, z).clamp(min=self.eps)
        out = num / den.unsqueeze(-1)                               # (b,h,n,d)

        context = out.transpose(1, 2).contiguous()                  # (b,n,h,d)
        context = context.view(b, context.shape[1], self.all_head_size)
        return context, None


class PerformerViTSelfAttention(ViTSelfAttention):
    """HF ViTSelfAttention with the THIRD-PARTY performer-pytorch FastAttention
    (FAVOR+) as the core op -- the trusted Performer baseline, replacing our own
    (broken) FAVOR+. Reuses HF's q/k/v Linears (params identical); FastAttention
    is param-free, so param-match with softmax holds."""

    def __init__(self, config, nb_features):
        super().__init__(config)
        from performer_pytorch import FastAttention
        self.fast_attn = FastAttention(dim_heads=self.attention_head_size,
                                       nb_features=nb_features, causal=False)

    def forward(self, hidden_states, head_mask=None):
        b = hidden_states.shape[0]
        shape = b, -1, self.num_attention_heads, self.attention_head_size
        q = self.query(hidden_states).view(*shape).transpose(1, 2)   # (b,h,n,d)
        k = self.key(hidden_states).view(*shape).transpose(1, 2)
        v = self.value(hidden_states).view(*shape).transpose(1, 2)
        out = self.fast_attn(q, k, v)                               # (b,h,n,d)
        context = out.transpose(1, 2).contiguous().view(b, -1, self.all_head_size)
        return context, None


def build_hf_vit(variant: str, *, image_size=32, patch_size=4, num_classes=100,
                 hidden_size=192, depth=6, heads=3, mlp_dim=384,
                 num_features=256, bandwidth=None, layers=1, dc=1.0, qk_norm=True,
                 coupling=1.0, dropout=0.0) -> ViTForImageClassification:
    """Trusted HF ViT with `variant` attention. All variants are param-identical."""
    if variant not in HF_VARIANTS:
        raise ValueError(f"variant must be one of {HF_VARIANTS}")
    cfg = ViTConfig(
        image_size=image_size, patch_size=patch_size, num_channels=3,
        hidden_size=hidden_size, num_hidden_layers=depth, num_attention_heads=heads,
        intermediate_size=mlp_dim, num_labels=num_classes,
        hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
    )
    model = ViTForImageClassification(cfg)
    if variant == "softmax":
        return model                                                # HF, untouched

    dim_head = hidden_size // heads
    for layer in model.vit.encoder.layer:
        orig = layer.attention.attention                            # ViTSelfAttention
        if variant == "performer":                                  # trusted FAVOR+
            lin = PerformerViTSelfAttention(cfg, nb_features=num_features)
        else:
            fmap = _make_feature_map(variant, dim_head, num_features, bandwidth, layers,
                                     dc, qk_norm, coupling)
            lin = LinearViTSelfAttention(cfg, fmap)
        lin.load_state_dict(orig.state_dict(), strict=False)        # copy q/k/v init
        layer.attention.attention = lin
    return model


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def assert_param_parity(cfg: dict, variants=HF_VARIANTS) -> int:
    counts = {v: count_params(build_hf_vit(v, **cfg)) for v in variants}
    ref = counts["softmax"]
    bad = {v: c for v, c in counts.items() if c != ref}
    assert not bad, f"PARAM MISMATCH vs softmax={ref:,}: {bad}"
    return ref

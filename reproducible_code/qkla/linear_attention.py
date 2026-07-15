"""Linear attention with a pluggable feature map phi.

Mirrors vit_pytorch/vit.py's Attention line-for-line -- same LayerNorm, same
to_qkv, same to_out, same head reshape -- so a softmax ViT and a linear-attn
ViT have IDENTICAL parameter counts. The ONLY difference is the core op:

    softmax:  out = softmax(q k^T / sqrt(d)) v          O(n^2 d)
    linear:   out = phi(q) (phi(k)^T v) / norm          O(n d r)

The feature map carries no trainable parameters (W is a buffer), so swapping
softmax -> linear -> quantum changes the inductive bias and the complexity
WITHOUT changing the parameter budget. That is what makes the comparison fair.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import Module
from einops import rearrange

from .feature_maps import FeatureMap, GaussianRFF


class LinearAttention(Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.,
                 feature_map: FeatureMap | None = None, num_features: int = 256,
                 eps: float = 1e-6):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.eps = eps

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        # default control feature map = generic Gaussian random features
        self.feature_map = feature_map or GaussianRFF(dim_head, num_features)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        phi_q = self.feature_map(q, is_query=True)      # (b, h, n, r)
        phi_k = self.feature_map(k, is_query=False)     # (b, h, n, r)

        # linear-attention reassociation: (phi_q) (phi_k^T v)
        kv = torch.einsum('b h n r, b h n d -> b h r d', phi_k, v)
        num = torch.einsum('b h n r, b h r d -> b h n d', phi_q, kv)

        z = phi_k.sum(dim=2)                            # (b, h, r)
        den = torch.einsum('b h n r, b h r -> b h n', phi_q, z).clamp(min=self.eps)
        out = num / den.unsqueeze(-1)

        out = self.dropout(out)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

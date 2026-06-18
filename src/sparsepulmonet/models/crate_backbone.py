from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


def pair(value):
    return value if isinstance(value, tuple) else (value, value)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class ISTABlock(nn.Module):
    """ISTA-inspired sparse-coding update used by CRATE.

    This is adapted from the public CRATE implementation. It is retained as the
    theory-guided image encoder component of SparsePulmoNet.
    """

    def __init__(self, dim: int, step_size: float = 0.1, lambd: float = 0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dim))
        init.kaiming_uniform_(self.weight)
        self.step_size = step_size
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = F.linear(x, self.weight, bias=None)
        grad_1 = F.linear(x1, self.weight.t(), bias=None)
        grad_2 = F.linear(x, self.weight.t(), bias=None)
        grad_update = self.step_size * (grad_2 - grad_1) - self.step_size * self.lambd
        return F.relu(x + grad_update)


class CRATEAttention(nn.Module):
    """Multi-head subspace self-attention style block.

    CRATE uses one learned projection per head, shared across query/key/value;
    this is intentionally different from a standard ViT qkv triplet.
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.qkv = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if not (heads == 1 and dim_head == dim)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        w = rearrange(self.qkv(x), "b n (h d) -> b h n d", h=self.heads)
        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, w)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        if return_attention:
            return out, attn, w
        return out


class CRATETransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        dropout: float = 0.0,
        ista: float = 0.1,
        ista_lambda: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PreNorm(dim, CRATEAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, ISTABlock(dim, step_size=ista, lambd=ista_lambda)),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.depth = depth
        self.heads = heads
        self.dim = dim
        self.dim_head = dim_head

    def forward(self, x: torch.Tensor, return_layerwise: bool = False):
        layerwise: List[Dict[str, torch.Tensor]] = []
        for layer_idx, (attn, ff) in enumerate(self.layers):
            mssa_out = attn(x)
            z_half = mssa_out + x
            z_next = ff(z_half)
            if return_layerwise:
                layerwise.append(
                    {
                        "layer": torch.tensor(layer_idx, device=x.device),
                        "z_half": z_half,
                        "z_next": z_next,
                    }
                )
            x = z_next
        if return_layerwise:
            return x, layerwise
        return x


class CRATEBackbone(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        dim: int,
        depth: int,
        heads: int,
        pool: str = "cls",
        channels: int = 1,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        ista: float = 0.1,
        ista_lambda: float = 0.1,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        if image_height % patch_height != 0 or image_width % patch_width != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if pool not in {"cls", "mean"}:
            raise ValueError("pool must be 'cls' or 'mean'")

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = CRATETransformer(dim, depth, heads, dim_head, dropout, ista=ista, ista_lambda=ista_lambda)
        self.pool = pool
        self.feature_dim = dim
        self.depth = depth

    def tokens(self, image: torch.Tensor) -> torch.Tensor:
        x = self.to_patch_embedding(image)
        batch_size, num_tokens, _ = x.shape
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=batch_size)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :(num_tokens + 1)]
        return self.dropout(x)

    def forward_features(self, image: torch.Tensor, return_layerwise: bool = False):
        x = self.tokens(image)
        if return_layerwise:
            x, layerwise = self.transformer(x, return_layerwise=True)
        else:
            x = self.transformer(x)
            layerwise = None
        pooled = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        if return_layerwise:
            return pooled, layerwise
        return pooled

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.forward_features(image)


def crate_tiny(**kwargs) -> CRATEBackbone:
    return CRATEBackbone(dim=384, depth=12, heads=6, dim_head=384 // 6, **kwargs)


def crate_small(**kwargs) -> CRATEBackbone:
    return CRATEBackbone(dim=576, depth=12, heads=12, dim_head=576 // 12, **kwargs)


def crate_base(**kwargs) -> CRATEBackbone:
    return CRATEBackbone(dim=768, depth=12, heads=12, dim_head=768 // 12, **kwargs)


def crate_large(**kwargs) -> CRATEBackbone:
    return CRATEBackbone(dim=1024, depth=24, heads=16, dim_head=1024 // 16, **kwargs)


BACKBONE_FACTORY = {
    "crate_tiny": crate_tiny,
    "crate_small": crate_small,
    "crate_base": crate_base,
    "crate_large": crate_large,
}

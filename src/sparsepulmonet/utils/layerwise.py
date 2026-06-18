from __future__ import annotations

from typing import Dict, Iterable, List

import torch


def coding_rate(tokens: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
    """Compute a CRATE-style coding-rate proxy for tokens of shape [B, N, D]."""
    if tokens.dim() != 3:
        raise ValueError(f"Expected tokens with shape [B, N, D], got {tuple(tokens.shape)}")
    x = tokens / torch.clamp(torch.norm(tokens, dim=-1, keepdim=True), min=1e-12)
    b, n, d = x.shape
    product = torch.matmul(x, x.transpose(-1, -2))
    eye = torch.eye(n, device=x.device, dtype=x.dtype).unsqueeze(0).expand(b, n, n)
    scalar = d / max(n * eps, 1e-12)
    sign, logabsdet = torch.linalg.slogdet(eye + scalar * product)
    values = 0.5 * logabsdet
    values = torch.where(sign > 0, values, torch.full_like(values, float("nan")))
    return values


def _nanmean_float(x: torch.Tensor) -> float:
    valid = x[~torch.isnan(x)]
    if valid.numel() == 0:
        return float("nan")
    return float(valid.mean().detach().cpu().item())


def _nanstd_float(x: torch.Tensor) -> float:
    valid = x[~torch.isnan(x)]
    if valid.numel() <= 1:
        return 0.0
    return float(valid.std(unbiased=False).detach().cpu().item())


def sparsity_stats(tokens: torch.Tensor, threshold: float = 1e-6) -> Dict[str, float]:
    abs_tokens = torch.abs(tokens)
    zero_fraction = (abs_tokens <= threshold).float().mean()
    nonzero_fraction = (abs_tokens > threshold).float().mean()
    return {
        "zero_fraction": float(zero_fraction.detach().cpu().item()),
        "nonzero_fraction": float(nonzero_fraction.detach().cpu().item()),
    }


@torch.no_grad()
def summarize_layerwise(layerwise: Iterable[Dict[str, torch.Tensor]], eps: float = 0.01, threshold: float = 1e-6) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for idx, record in enumerate(layerwise):
        z_half = record["z_half"]
        z_next = record["z_next"]
        cr = coding_rate(z_half, eps=eps)
        sp = sparsity_stats(z_next, threshold=threshold)
        rows.append(
            {
                "layer": idx + 1,
                "coding_rate_mean": _nanmean_float(cr),
                "coding_rate_std": _nanstd_float(cr),
                **sp,
            }
        )
    return rows

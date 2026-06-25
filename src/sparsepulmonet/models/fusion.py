from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


def _concat(image_feat: torch.Tensor, tabular_feat: torch.Tensor) -> torch.Tensor:
    if tabular_feat.numel() == 0 or tabular_feat.shape[1] == 0:
        return image_feat
    return torch.cat([image_feat, tabular_feat], dim=1)


class ConcatFusion(nn.Module):
    """Direct concatenation fusion used for Table 5 ablation."""

    def __init__(self, image_dim: int, tabular_dim: int):
        super().__init__()
        self.image_dim = image_dim
        self.tabular_dim = tabular_dim
        self.output_dim = image_dim + tabular_dim

    def forward(self, image_feat: torch.Tensor, tabular_feat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        combined = _concat(image_feat, tabular_feat)
        aux = {
            "combined": combined,
            "fused": combined,
            "mu": torch.zeros(image_feat.shape[0], 1, device=image_feat.device, dtype=image_feat.dtype),
            "logvar": torch.zeros(image_feat.shape[0], 1, device=image_feat.device, dtype=image_feat.dtype),
        }
        return combined, aux


class AttentionFusion(nn.Module):
    """Simple tabular-gated image fusion used for Table 5 ablation."""

    def __init__(self, image_dim: int, tabular_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.image_dim = image_dim
        self.tabular_dim = tabular_dim
        self.output_dim = image_dim + tabular_dim
        if tabular_dim > 0:
            self.gate = nn.Sequential(
                nn.Linear(tabular_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, image_dim),
                nn.Sigmoid(),
            )
        else:
            self.gate = None

    def forward(self, image_feat: torch.Tensor, tabular_feat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.gate is None or tabular_feat.numel() == 0 or tabular_feat.shape[1] == 0:
            fused = image_feat
            gate = torch.ones_like(image_feat)
        else:
            gate = self.gate(tabular_feat)
            fused_img = image_feat * (1.0 + gate)
            fused = torch.cat([fused_img, tabular_feat], dim=1)
        aux = {
            "combined": _concat(image_feat, tabular_feat),
            "fused": fused,
            "gate": gate,
            "mu": torch.zeros(image_feat.shape[0], 1, device=image_feat.device, dtype=image_feat.dtype),
            "logvar": torch.zeros(image_feat.shape[0], 1, device=image_feat.device, dtype=image_feat.dtype),
        }
        return fused, aux


class TabularVAEFusion(nn.Module):
    """VAE-style multimodal fusion of image, clinical, and radiomics features."""

    def __init__(self, image_dim: int, tabular_dim: int, latent_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.image_dim = image_dim
        self.tabular_dim = tabular_dim
        self.input_dim = image_dim + tabular_dim
        self.output_dim = self.input_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.input_dim),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * torch.clamp(logvar, min=-20.0, max=20.0))
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, image_feat: torch.Tensor, tabular_feat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        combined = _concat(image_feat, tabular_feat)
        hidden = self.encoder(combined)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        aux = {
            "combined": combined,
            "fused": reconstructed,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }
        return reconstructed, aux


def build_fusion(fusion_type: str, image_dim: int, tabular_dim: int, latent_dim: int = 32, hidden_dim: int = 256) -> nn.Module:
    if fusion_type == "vae":
        return TabularVAEFusion(image_dim=image_dim, tabular_dim=tabular_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
    if fusion_type == "concat":
        return ConcatFusion(image_dim=image_dim, tabular_dim=tabular_dim)
    if fusion_type == "attention":
        return AttentionFusion(image_dim=image_dim, tabular_dim=tabular_dim, hidden_dim=hidden_dim)
    if fusion_type == "image_only":
        return ConcatFusion(image_dim=image_dim, tabular_dim=0)
    raise ValueError(f"Unknown fusion_type: {fusion_type}")

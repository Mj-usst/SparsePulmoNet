from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


class TabularVAEFusion(nn.Module):
    """VAE-style multimodal fusion of image, clinical, and radiomics features."""

    def __init__(self, image_dim: int, tabular_dim: int, latent_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.image_dim = image_dim
        self.tabular_dim = tabular_dim
        self.input_dim = image_dim + tabular_dim
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
        combined = torch.cat([image_feat, tabular_feat], dim=1)
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

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from sparsepulmonet.config import ModelConfig
from sparsepulmonet.models.crate_backbone import BACKBONE_FACTORY
from sparsepulmonet.models.fusion import TabularVAEFusion


class SparsePulmoNet(nn.Module):
    """CRATE image encoder + clinical/radiomics VAE fusion for IPF FVC-slope prediction."""

    def __init__(self, cfg: ModelConfig, tabular_dim: int):
        super().__init__()
        if cfg.model_name not in BACKBONE_FACTORY:
            raise ValueError(f"Unknown model_name: {cfg.model_name}")
        self.cfg = cfg
        self.backbone = BACKBONE_FACTORY[cfg.model_name](
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            channels=cfg.in_channels,
            dropout=cfg.dropout,
            emb_dropout=cfg.emb_dropout,
            ista=cfg.ista_step_size,
            ista_lambda=cfg.ista_lambda,
            pool=cfg.pool,
        )
        fused_dim = self.backbone.feature_dim + tabular_dim
        self.fusion = TabularVAEFusion(
            image_dim=self.backbone.feature_dim,
            tabular_dim=tabular_dim,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.fusion_hidden_dim,
        )
        self.slope_head = nn.Sequential(nn.LayerNorm(fused_dim), nn.Linear(fused_dim, 1))
        self.predict_sigma = cfg.predict_sigma
        if cfg.predict_sigma:
            self.sigma_head = nn.Sequential(nn.LayerNorm(fused_dim), nn.Linear(fused_dim, 1), nn.Softplus())
        else:
            self.sigma_head = None

    def forward(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor,
        return_layerwise: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if return_layerwise:
            image_feat, layerwise = self.backbone.forward_features(image, return_layerwise=True)
        else:
            image_feat = self.backbone.forward_features(image)
            layerwise = None
        fused, aux = self.fusion(image_feat, tabular)
        slope = self.slope_head(fused).squeeze(-1)
        aux["image_feat"] = image_feat
        if layerwise is not None:
            aux["layerwise"] = layerwise
        if self.sigma_head is not None:
            aux["sigma"] = self.sigma_head(fused).squeeze(-1) + 70.0
        return slope, aux


# Backward-compatible alias for older internal scripts.
CRATEIPFRegressor = SparsePulmoNet

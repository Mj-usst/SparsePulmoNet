from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from sparsepulmonet.config import ModelConfig
from sparsepulmonet.models.baseline_backbones import BASELINE_BACKBONE_FACTORY
from sparsepulmonet.models.crate_backbone import BACKBONE_FACTORY
from sparsepulmonet.models.fusion import build_fusion


class SparsePulmoNet(nn.Module):
    """Image encoder + optional clinical/radiomics fusion for IPF FVC-slope prediction.

    The default is the manuscript model: CRATE image encoder + VAE-style
    multimodal fusion. The same wrapper also supports Table 3 input ablations
    and Table 5 fusion/backbone ablations.
    """

    def __init__(self, cfg: ModelConfig, tabular_dim: int):
        super().__init__()
        self.cfg = cfg
        if cfg.model_name in BACKBONE_FACTORY:
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
        elif cfg.model_name in BASELINE_BACKBONE_FACTORY:
            self.backbone = BASELINE_BACKBONE_FACTORY[cfg.model_name](image_size=cfg.image_size, channels=cfg.in_channels)
        else:
            valid = sorted(list(BACKBONE_FACTORY.keys()) + list(BASELINE_BACKBONE_FACTORY.keys()))
            raise ValueError(f"Unknown model_name: {cfg.model_name}. Valid options: {valid}")

        effective_tabular_dim = 0 if cfg.fusion_type == "image_only" else tabular_dim
        self.fusion = build_fusion(
            fusion_type=cfg.fusion_type,
            image_dim=self.backbone.feature_dim,
            tabular_dim=effective_tabular_dim,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.fusion_hidden_dim,
        )
        self.slope_head = nn.Sequential(nn.LayerNorm(self.fusion.output_dim), nn.Linear(self.fusion.output_dim, 1))
        self.predict_sigma = cfg.predict_sigma
        if cfg.predict_sigma:
            self.sigma_head = nn.Sequential(nn.LayerNorm(self.fusion.output_dim), nn.Linear(self.fusion.output_dim, 1), nn.Softplus())
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

        if self.cfg.fusion_type == "image_only":
            tabular = tabular[:, :0]
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

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sparsepulmonet.config import ModelConfig
from sparsepulmonet.models.ipf_model import SparsePulmoNet
from sparsepulmonet.utils.layerwise import summarize_layerwise


def test_forward_pass():
    cfg = ModelConfig(model_name="crate_tiny", image_size=64, patch_size=16, latent_dim=8, fusion_hidden_dim=32)
    model = SparsePulmoNet(cfg, tabular_dim=13)
    image = torch.randn(2, 1, 64, 64)
    tabular = torch.randn(2, 13)
    slope, aux = model(image, tabular, return_layerwise=True)
    assert slope.shape == (2,)
    assert "fused" in aux and "combined" in aux and "layerwise" in aux
    rows = summarize_layerwise(aux["layerwise"])
    assert len(rows) == 12
    assert set(["layer", "coding_rate_mean", "zero_fraction", "nonzero_fraction"]).issubset(rows[0].keys())


if __name__ == "__main__":
    test_forward_pass()
    print("ok")

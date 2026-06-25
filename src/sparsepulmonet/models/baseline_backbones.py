from __future__ import annotations

from typing import Callable, Dict

import torch
from torch import nn
from torchvision import models


class TorchVisionBackbone(nn.Module):
    """TorchVision CNN backbone wrapper for Table 5 backbone ablations."""

    def __init__(self, model: nn.Module, feature_dim: int, repeat_channels: bool = True):
        super().__init__()
        self.model = model
        self.feature_dim = feature_dim
        self.repeat_channels = repeat_channels

    def forward_features(self, image: torch.Tensor, return_layerwise: bool = False):
        if image.shape[1] == 1 and self.repeat_channels:
            image = image.repeat(1, 3, 1, 1)
        feat = self.model(image)
        if feat.dim() > 2:
            feat = torch.flatten(feat, 1)
        if return_layerwise:
            return feat, []
        return feat

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.forward_features(image)


def _resnet18(**kwargs) -> TorchVisionBackbone:
    net = models.resnet18(weights=None)
    feature_dim = net.fc.in_features
    net.fc = nn.Identity()
    return TorchVisionBackbone(net, feature_dim=feature_dim)


def _resnet50(**kwargs) -> TorchVisionBackbone:
    net = models.resnet50(weights=None)
    feature_dim = net.fc.in_features
    net.fc = nn.Identity()
    return TorchVisionBackbone(net, feature_dim=feature_dim)


def _resnext50_32x4d(**kwargs) -> TorchVisionBackbone:
    net = models.resnext50_32x4d(weights=None)
    feature_dim = net.fc.in_features
    net.fc = nn.Identity()
    return TorchVisionBackbone(net, feature_dim=feature_dim)


def _efficientnet_b0(**kwargs) -> TorchVisionBackbone:
    net = models.efficientnet_b0(weights=None)
    feature_dim = net.classifier[1].in_features
    net.classifier = nn.Identity()
    return TorchVisionBackbone(net, feature_dim=feature_dim)


BASELINE_BACKBONE_FACTORY: Dict[str, Callable[..., TorchVisionBackbone]] = {
    "resnet18": _resnet18,
    "resnet50": _resnet50,
    "resnext50_32x4d": _resnext50_32x4d,
    "efficientnet_b0": _efficientnet_b0,
}

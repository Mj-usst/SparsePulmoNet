from __future__ import annotations

from typing import Dict, Mapping

import torch
from torch import nn
from tqdm.auto import tqdm

from sparsepulmonet.training.metrics import summarize_patient_predictions


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    logvar = torch.clamp(logvar, min=-20.0, max=20.0)
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def reconstruction_loss(reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(reconstructed - target))


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    slope_loss_weight: float = 1.0,
    recon_weight: float = 1.0,
    kl_weight: float = 1e-4,
    desc: str = "train",
) -> Dict[str, float]:
    model.train()
    reg_loss_fn = nn.L1Loss()
    totals = {"loss": 0.0, "slope_loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0}
    num_batches = 0

    for images, tabular, targets, _metas in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            slope_pred, aux = model(images, tabular)
            slope_loss = reg_loss_fn(slope_pred, targets)
            recon = reconstruction_loss(aux["fused"], aux["combined"])
            kl = kl_divergence(aux["mu"], aux["logvar"])
            loss = slope_loss_weight * slope_loss + recon_weight * recon + kl_weight * kl

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        totals["loss"] += float(loss.item())
        totals["slope_loss"] += float(slope_loss.item())
        totals["recon_loss"] += float(recon.item())
        totals["kl_loss"] += float(kl.item())
        num_batches += 1

    return {k: v / max(1, num_batches) for k, v in totals.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    val_patients,
    train_df,
    device: torch.device,
    use_amp: bool,
    fixed_sigma: float = 70.0,
    desc: str = "val",
) -> Dict[str, float]:
    model.eval()
    predictions_by_patient: Dict[str, float] = {}
    meta_by_patient: Dict[str, Mapping[str, float]] = {}
    slope_loss_fn = nn.L1Loss()
    total_slope_loss = 0.0
    num_batches = 0

    for images, tabular, targets, metas in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            slope_pred, _aux = model(images, tabular)
            slope_loss = slope_loss_fn(slope_pred, targets)
        total_slope_loss += float(slope_loss.item())
        num_batches += 1
        pred_np = slope_pred.detach().cpu().numpy().reshape(-1)
        for idx, meta in enumerate(metas):
            patient_id = meta["patient_id"]
            predictions_by_patient[patient_id] = float(pred_np[idx])
            meta_by_patient[patient_id] = {
                "baseline_week": float(meta["baseline_week"]),
                "baseline_fvc": float(meta["baseline_fvc"]),
            }

    summary = summarize_patient_predictions(
        predictions_by_patient,
        val_patients,
        train_df,
        patient_meta=meta_by_patient,
        sigma=fixed_sigma,
    )
    summary["slope_loss"] = total_slope_loss / max(1, num_batches)
    return summary

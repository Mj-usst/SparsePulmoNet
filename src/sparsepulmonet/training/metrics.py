from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


def reconstruct_fvc(
    patient_df: pd.DataFrame,
    slope_pred: float,
    baseline_week: Optional[float] = None,
    baseline_fvc: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    patient_df = patient_df.sort_values("Weeks")
    weeks = patient_df["Weeks"].to_numpy(dtype=np.float64)
    fvc_true = patient_df["FVC"].to_numpy(dtype=np.float64)
    if baseline_week is None or baseline_fvc is None:
        idx = int(np.argmin(np.abs(weeks)))
        baseline_week = float(weeks[idx])
        baseline_fvc = float(fvc_true[idx])
    fvc_pred = float(baseline_fvc) + float(slope_pred) * (weeks - float(baseline_week))
    return weeks, fvc_true, fvc_pred.astype(np.float64)


def laplace_log_likelihood(fvc_true: np.ndarray, fvc_pred: np.ndarray, sigma: np.ndarray | float = 70.0) -> float:
    """Modified Laplace log likelihood used by the OSIC benchmark.

    Values are negative; less negative / closer to zero is better.
    """
    sigma_arr = np.asarray(sigma, dtype=np.float64) + np.zeros_like(fvc_true, dtype=np.float64)
    sigma_clip = np.maximum(sigma_arr, 70.0)
    delta = np.minimum(np.abs(np.asarray(fvc_true) - np.asarray(fvc_pred)), 1000.0)
    metric = -np.sqrt(2.0) * delta / sigma_clip - np.log(np.sqrt(2.0) * sigma_clip)
    return float(np.mean(metric))


def patient_metrics(
    patient_id: str,
    slope_pred: float,
    train_df: pd.DataFrame,
    baseline_week: Optional[float] = None,
    baseline_fvc: Optional[float] = None,
    sigma: float = 70.0,
) -> Dict[str, float]:
    patient_df = train_df.loc[train_df["Patient"] == patient_id]
    weeks, fvc_true, fvc_pred = reconstruct_fvc(patient_df, slope_pred, baseline_week, baseline_fvc)
    err = fvc_true - fvc_pred
    return {
        "lllm": laplace_log_likelihood(fvc_true, fvc_pred, sigma=sigma),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "n_visits": int(len(fvc_true)),
    }


def summarize_patient_predictions(
    predictions: Mapping[str, float],
    patient_ids: Iterable[str],
    train_df: pd.DataFrame,
    patient_meta: Optional[Mapping[str, Mapping[str, float]]] = None,
    sigma: float = 70.0,
) -> Dict[str, float]:
    all_true = []
    all_pred = []
    patient_lllm = []
    patient_rmse = []
    patient_mae = []

    for pid in patient_ids:
        if pid not in predictions:
            continue
        meta = patient_meta.get(pid, {}) if patient_meta is not None else {}
        patient_df = train_df.loc[train_df["Patient"] == pid]
        _weeks, fvc_true, fvc_pred = reconstruct_fvc(
            patient_df,
            predictions[pid],
            baseline_week=meta.get("baseline_week"),
            baseline_fvc=meta.get("baseline_fvc"),
        )
        all_true.append(fvc_true)
        all_pred.append(fvc_pred)
        err = fvc_true - fvc_pred
        patient_lllm.append(laplace_log_likelihood(fvc_true, fvc_pred, sigma=sigma))
        patient_rmse.append(float(np.sqrt(np.mean(err**2))))
        patient_mae.append(float(np.mean(np.abs(err))))

    if not all_true:
        return {"lllm": float("nan"), "rmse": float("nan"), "mae": float("nan"), "n_patients": 0, "n_visits": 0}

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    err = y_true - y_pred
    return {
        "lllm": laplace_log_likelihood(y_true, y_pred, sigma=sigma),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "patient_mean_lllm": float(np.mean(patient_lllm)),
        "patient_mean_rmse": float(np.mean(patient_rmse)),
        "patient_mean_mae": float(np.mean(patient_mae)),
        "n_patients": int(len(patient_lllm)),
        "n_visits": int(len(y_true)),
    }

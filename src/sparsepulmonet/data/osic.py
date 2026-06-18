from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from sparsepulmonet.config import DataConfig


@dataclass
class FoldNormalizers:
    age_mean: float
    age_std: float
    radiomics_mean: np.ndarray
    radiomics_std: np.ndarray


@dataclass
class PatientRecord:
    patient_id: str
    slope: float
    baseline_week: float
    baseline_fvc: float
    tabular: np.ndarray


def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"Could not find any of these columns: {list(candidates)}")


def read_osic_train_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"OSIC train.csv is missing required columns: {sorted(missing)}")
    return df


def read_radiomics_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    patient_col = _first_existing_column(df, ["Patient", "patient", "PatientID", "patient_id", "ID"])
    if patient_col != "Patient":
        df = df.rename(columns={patient_col: "Patient"})
    return df


def fit_patient_slope(patient_df: pd.DataFrame) -> Tuple[float, float]:
    """Fit FVC = slope * Week + intercept by ordinary least squares."""
    weeks = patient_df["Weeks"].to_numpy(dtype=np.float64)
    fvc = patient_df["FVC"].to_numpy(dtype=np.float64)
    design = np.vstack([weeks, np.ones_like(weeks)]).T
    slope, intercept = np.linalg.lstsq(design, fvc, rcond=None)[0]
    return float(slope), float(intercept)


def baseline_from_closest_week0(patient_df: pd.DataFrame) -> Tuple[float, float]:
    idx = np.argmin(np.abs(patient_df["Weeks"].to_numpy(dtype=np.float64)))
    row = patient_df.iloc[int(idx)]
    return float(row["Weeks"]), float(row["FVC"])


def fit_normalizers(train_df: pd.DataFrame, radiomics_df: pd.DataFrame, train_patients: Sequence[str], cfg: DataConfig) -> FoldNormalizers:
    train_rows = train_df.loc[train_df["Patient"].isin(train_patients)]
    age_mean = float(train_rows["Age"].mean())
    age_std = float(train_rows["Age"].std(ddof=0) or 1.0)

    rad_rows = radiomics_df.loc[radiomics_df["Patient"].isin(train_patients), cfg.radiomics_columns]
    rad_values = rad_rows.to_numpy(dtype=np.float32)
    rad_mean = np.nanmean(rad_values, axis=0).astype(np.float32)
    rad_std = np.nanstd(rad_values, axis=0).astype(np.float32)
    rad_std[rad_std == 0] = 1.0
    return FoldNormalizers(age_mean=age_mean, age_std=age_std, radiomics_mean=rad_mean, radiomics_std=rad_std)


def build_clinical_vector(patient_df: pd.DataFrame, normalizers: FoldNormalizers) -> List[float]:
    age = float(patient_df["Age"].iloc[0])
    age_norm = (age - normalizers.age_mean) / normalizers.age_std
    sex = 0.0 if str(patient_df["Sex"].iloc[0]).lower() == "male" else 1.0

    smoking = str(patient_df["SmokingStatus"].iloc[0]).strip().lower()
    if smoking == "never smoked":
        smoke = [0.0, 0.0]
    elif smoking in {"ex-smoker", "ex smoker", "former smoker"}:
        smoke = [1.0, 1.0]
    elif smoking in {"currently smokes", "current smoker", "currently smoking"}:
        smoke = [0.0, 1.0]
    else:
        smoke = [1.0, 0.0]
    return [float(age_norm), float(sex), float(smoke[0]), float(smoke[1])]


def build_patient_records(
    train_df: pd.DataFrame,
    radiomics_df: pd.DataFrame,
    patient_ids: Sequence[str],
    normalizers: FoldNormalizers,
    cfg: DataConfig,
) -> Dict[str, PatientRecord]:
    records: Dict[str, PatientRecord] = {}
    missing_radiomics: List[str] = []
    missing_columns = [col for col in cfg.radiomics_columns if col not in radiomics_df.columns]
    if missing_columns:
        raise ValueError(f"Radiomics CSV is missing selected feature columns: {missing_columns}")

    excluded = set(cfg.exclude_patient_ids)
    for patient_id in patient_ids:
        if patient_id in excluded:
            continue
        patient_df = train_df.loc[train_df["Patient"] == patient_id].sort_values("Weeks")
        if patient_df.empty:
            continue

        radiomics_row = radiomics_df.loc[radiomics_df["Patient"] == patient_id]
        if radiomics_row.empty:
            missing_radiomics.append(patient_id)
            continue

        slope, _intercept = fit_patient_slope(patient_df)
        baseline_week, baseline_fvc = baseline_from_closest_week0(patient_df)
        clinical = build_clinical_vector(patient_df, normalizers)
        radiomics = radiomics_row[cfg.radiomics_columns].iloc[0].to_numpy(dtype=np.float32)
        if cfg.normalize_radiomics_per_fold:
            radiomics = (radiomics - normalizers.radiomics_mean) / normalizers.radiomics_std
        radiomics = np.nan_to_num(radiomics, nan=0.0, posinf=0.0, neginf=0.0)
        tabular = np.asarray(clinical + radiomics.astype(np.float32).tolist(), dtype=np.float32)
        records[patient_id] = PatientRecord(
            patient_id=patient_id,
            slope=slope,
            baseline_week=baseline_week,
            baseline_fvc=baseline_fvc,
            tabular=tabular,
        )

    if missing_radiomics and cfg.strict:
        preview = ", ".join(missing_radiomics[:10])
        raise ValueError(
            f"Missing radiomics rows for {len(missing_radiomics)} patients. First missing IDs: {preview}. "
            "Either provide a complete radiomics CSV or run with strict=False."
        )
    return records


def _natural_key(path: Path) -> Tuple:
    parts = re.split(r"(\d+)", path.stem)
    return tuple(int(p) if p.isdigit() else p for p in parts)


def list_dicom_files(patient_dir: Path) -> List[Path]:
    if not patient_dir.exists():
        return []
    files = [p for p in patient_dir.iterdir() if p.is_file() and not p.name.startswith(".")]
    if not files:
        return []

    sortable = []
    for p in files:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            z = getattr(ds, "ImagePositionPatient", None)
            z_value = float(z[2]) if z is not None and len(z) >= 3 else None
            inst = getattr(ds, "InstanceNumber", None)
            key = (0, z_value) if z_value is not None else (1, int(inst)) if inst is not None else (2, _natural_key(p))
        except Exception:
            key = (2, _natural_key(p))
        sortable.append((key, p))
    return [p for _key, p in sorted(sortable, key=lambda item: item[0])]


def candidate_middle_slices(slices: Sequence[Path], strip_ratio: float) -> List[Path]:
    if not slices:
        return []
    strip_ratio = min(max(float(strip_ratio), 0.0), 0.49)
    start = int(len(slices) * strip_ratio)
    end = int(len(slices) * (1.0 - strip_ratio))
    clipped = list(slices[start:end]) if end > start else list(slices)
    return clipped or list(slices)


def read_dicom_image(path: Path, cfg: DataConfig) -> np.ndarray:
    dcm = pydicom.dcmread(str(path), force=True)
    image = dcm.pixel_array.astype(np.float32)
    slope = float(getattr(dcm, "RescaleSlope", 1.0))
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
    image = image * slope + intercept

    if str(getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")).upper() == "MONOCHROME1":
        image = np.max(image) - image

    lower = cfg.window_level - cfg.window_width / 2.0
    upper = cfg.window_level + cfg.window_width / 2.0
    image = np.clip(image, lower, upper)
    image = (image - lower) / max(upper - lower, 1e-6)

    if cfg.use_median_filter:
        image = cv2.medianBlur((image * 255).astype(np.uint8), 3).astype(np.float32) / 255.0

    image = cv2.resize(image, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_LINEAR)
    return image.astype(np.float32)


class OSICPatientDataset(Dataset):
    def __init__(
        self,
        patient_ids: Iterable[str],
        records: Mapping[str, PatientRecord],
        cfg: DataConfig,
        mode: str,
    ):
        self.patient_ids = list(patient_ids)
        self.records = records
        self.cfg = cfg
        self.mode = mode
        self.patient_slices: Dict[str, List[Path]] = {}
        missing_dirs: List[str] = []

        for patient_id in self.patient_ids:
            patient_dir = cfg.train_dir / patient_id
            slices = candidate_middle_slices(list_dicom_files(patient_dir), cfg.strip_ratio)
            if not slices:
                missing_dirs.append(patient_id)
                continue
            self.patient_slices[patient_id] = slices

        self.patient_ids = [pid for pid in self.patient_ids if pid in self.patient_slices and pid in self.records]
        if missing_dirs and cfg.strict:
            preview = ", ".join(missing_dirs[:10])
            raise FileNotFoundError(f"Missing/empty DICOM folders for {len(missing_dirs)} patients. First IDs: {preview}")

    def __len__(self) -> int:
        return len(self.patient_ids)

    def _select_slice(self, patient_id: str) -> Path:
        slices = self.patient_slices[patient_id]
        policy = self.cfg.train_slice_policy if self.mode == "train" else self.cfg.eval_slice_policy
        if policy == "random_middle":
            return Path(np.random.choice(slices))
        return slices[len(slices) // 2]

    def __getitem__(self, idx: int):
        patient_id = self.patient_ids[idx]
        record = self.records[patient_id]
        dicom_path = self._select_slice(patient_id)
        image = read_dicom_image(dicom_path, self.cfg)
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        tabular_tensor = torch.from_numpy(record.tabular).float()
        target_tensor = torch.tensor(record.slope, dtype=torch.float32)
        meta = {
            "patient_id": patient_id,
            "baseline_week": record.baseline_week,
            "baseline_fvc": record.baseline_fvc,
            "slice_path": str(dicom_path),
        }
        return image_tensor, tabular_tensor, target_tensor, meta


def _collate(batch):
    images, tabular, targets, metas = zip(*batch)
    return torch.stack(images), torch.stack(tabular), torch.stack(targets), list(metas)


def make_osic_folds(cfg: DataConfig):
    train_df = read_osic_train_csv(cfg.train_csv)
    radiomics_df = read_radiomics_csv(cfg.radiomics_csv)
    all_patients = sorted([p for p in train_df["Patient"].unique().tolist() if p not in set(cfg.exclude_patient_ids)])
    kfold = KFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.random_state)
    folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(all_patients)):
        train_patients = [all_patients[i] for i in train_idx]
        val_patients = [all_patients[i] for i in val_idx]
        normalizers = fit_normalizers(train_df, radiomics_df, train_patients, cfg)
        records = build_patient_records(train_df, radiomics_df, train_patients + val_patients, normalizers, cfg)
        train_dataset = OSICPatientDataset(train_patients, records, cfg, mode="train")
        val_dataset = OSICPatientDataset(val_patients, records, cfg, mode="val")

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=_collate,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=max(1, min(cfg.batch_size, 8)),
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=_collate,
            drop_last=False,
        )
        folds.append(
            {
                "fold": fold_idx,
                "train_loader": train_loader,
                "val_loader": val_loader,
                "train_patients": train_patients,
                "val_patients": val_patients,
                "train_df": train_df,
                "records": records,
                "normalizers": normalizers,
                "summary": {
                    "fold": fold_idx,
                    "n_train_patients": len(train_dataset),
                    "n_val_patients": len(val_dataset),
                    "n_all_patients_in_csv": len(all_patients),
                    "excluded_patient_ids": list(cfg.exclude_patient_ids),
                },
            }
        )
    return folds


def write_fold_summary(folds: Sequence[Mapping], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([fold["summary"] for fold in folds], f, indent=2, ensure_ascii=False)

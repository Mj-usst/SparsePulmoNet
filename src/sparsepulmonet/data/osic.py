from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from sparsepulmonet.config import DataConfig
from sparsepulmonet.data.preprocessing import middle_slice_indices, preprocess_patient_volume


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
    age_mean = float(train_rows["Age"].mean()) if cfg.use_clinical else 0.0
    age_std = float(train_rows["Age"].std(ddof=0) or 1.0) if cfg.use_clinical else 1.0

    if cfg.use_radiomics:
        missing_columns = [col for col in cfg.radiomics_columns if col not in radiomics_df.columns]
        if missing_columns:
            raise ValueError(f"Radiomics CSV is missing selected feature columns: {missing_columns}")
        rad_rows = radiomics_df.loc[radiomics_df["Patient"].isin(train_patients), cfg.radiomics_columns]
        rad_values = rad_rows.to_numpy(dtype=np.float32)
        rad_mean = np.nanmean(rad_values, axis=0).astype(np.float32)
        rad_std = np.nanstd(rad_values, axis=0).astype(np.float32)
        rad_std[rad_std == 0] = 1.0
    else:
        rad_mean = np.zeros((0,), dtype=np.float32)
        rad_std = np.ones((0,), dtype=np.float32)
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
    excluded = set(cfg.exclude_patient_ids)

    for patient_id in patient_ids:
        if patient_id in excluded:
            continue
        patient_df = train_df.loc[train_df["Patient"] == patient_id].sort_values("Weeks")
        if patient_df.empty:
            continue

        slope, _intercept = fit_patient_slope(patient_df)
        baseline_week, baseline_fvc = baseline_from_closest_week0(patient_df)

        features: List[float] = []
        if cfg.use_clinical:
            features.extend(build_clinical_vector(patient_df, normalizers))

        if cfg.use_radiomics:
            radiomics_row = radiomics_df.loc[radiomics_df["Patient"] == patient_id]
            if radiomics_row.empty:
                missing_radiomics.append(patient_id)
                continue
            radiomics = radiomics_row[cfg.radiomics_columns].iloc[0].to_numpy(dtype=np.float32)
            if cfg.normalize_radiomics_per_fold:
                radiomics = (radiomics - normalizers.radiomics_mean) / normalizers.radiomics_std
            radiomics = np.nan_to_num(radiomics, nan=0.0, posinf=0.0, neginf=0.0)
            features.extend(radiomics.astype(np.float32).tolist())

        tabular = np.asarray(features, dtype=np.float32)
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
            "Either provide a complete radiomics CSV, disable radiomics, or run with strict=False."
        )
    return records


class OSICPatientDataset(Dataset):
    """Patient-level OSIC dataset with manuscript-aligned CT preprocessing.

    Each patient contributes one CT slice per epoch. Slices are selected from the
    preprocessed middle-lung volume: random middle slice for training and center
    middle slice for validation/evaluation.
    """

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
        if self.cfg.preprocessed_cache_dir is None:
            self.cfg.preprocessed_cache_dir = self.cfg.data_root / ".sparsepulmonet_cache"
        self.patient_slice_indices: Dict[str, List[int]] = {}
        missing_dirs: List[str] = []

        for patient_id in self.patient_ids:
            patient_dir = cfg.train_dir / patient_id
            if not patient_dir.exists():
                missing_dirs.append(patient_id)
                continue
            try:
                volume, _mask = preprocess_patient_volume(patient_dir, cfg)
                indices = middle_slice_indices(volume.shape[0], cfg.strip_ratio)
            except Exception as exc:
                if cfg.strict:
                    raise RuntimeError(f"Failed to preprocess patient {patient_id}: {exc}") from exc
                missing_dirs.append(patient_id)
                continue
            self.patient_slice_indices[patient_id] = indices

        self.patient_ids = [pid for pid in self.patient_ids if pid in self.patient_slice_indices and pid in self.records]
        if missing_dirs and cfg.strict:
            preview = ", ".join(missing_dirs[:10])
            raise FileNotFoundError(f"Missing/failed DICOM preprocessing for {len(missing_dirs)} patients. First IDs: {preview}")

    def __len__(self) -> int:
        return len(self.patient_ids)

    def _select_index(self, patient_id: str) -> int:
        indices = self.patient_slice_indices[patient_id]
        policy = self.cfg.train_slice_policy if self.mode == "train" else self.cfg.eval_slice_policy
        if policy == "random_middle":
            return int(np.random.choice(indices))
        return int(indices[len(indices) // 2])

    def __getitem__(self, idx: int):
        patient_id = self.patient_ids[idx]
        record = self.records[patient_id]
        patient_dir = self.cfg.train_dir / patient_id
        volume, _mask = preprocess_patient_volume(patient_dir, self.cfg)
        slice_idx = self._select_index(patient_id)
        image = volume[slice_idx]
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        tabular_tensor = torch.from_numpy(record.tabular).float()
        target_tensor = torch.tensor(record.slope, dtype=torch.float32)
        meta = {
            "patient_id": patient_id,
            "baseline_week": record.baseline_week,
            "baseline_fvc": record.baseline_fvc,
            "slice_index": slice_idx,
        }
        return image_tensor, tabular_tensor, target_tensor, meta


def _collate(batch):
    images, tabular, targets, metas = zip(*batch)
    return torch.stack(images), torch.stack(tabular), torch.stack(targets), list(metas)


def make_osic_folds(cfg: DataConfig):
    train_df = read_osic_train_csv(cfg.train_csv)
    radiomics_df = read_radiomics_csv(cfg.radiomics_csv) if cfg.use_radiomics else pd.DataFrame({"Patient": train_df["Patient"].unique()})
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
                    "use_clinical": cfg.use_clinical,
                    "use_radiomics": cfg.use_radiomics,
                    "use_lung_mask": cfg.use_lung_mask,
                    "enable_isotropic_resampling": cfg.enable_isotropic_resampling,
                    "excluded_patient_ids": list(cfg.exclude_patient_ids),
                },
            }
        )
    return folds


def write_fold_summary(folds: Sequence[Mapping], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([fold["summary"] for fold in folds], f, indent=2, ensure_ascii=False)

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import pydicom

from sparsepulmonet.config import DataConfig


def sort_dicom_files(files: Sequence[Path]) -> List[Path]:
    sortable = []
    for path in files:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            pos = getattr(ds, "ImagePositionPatient", None)
            z_value = float(pos[2]) if pos is not None and len(pos) >= 3 else None
            inst = getattr(ds, "InstanceNumber", None)
            key = (0, z_value) if z_value is not None else (1, int(inst)) if inst is not None else (2, path.name)
        except Exception:
            key = (2, path.name)
        sortable.append((key, path))
    return [p for _key, p in sorted(sortable, key=lambda item: item[0])]


def load_dicom_series(patient_dir: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Load a DICOM series and convert pixels to HU.

    Returns:
        volume_hu: array with shape [Z, H, W]
        spacing: tuple (z, y, x) in mm
    """
    files = sort_dicom_files([p for p in patient_dir.iterdir() if p.is_file() and not p.name.startswith(".")])
    if not files:
        raise FileNotFoundError(f"No DICOM files found in {patient_dir}")

    slices = []
    z_positions = []
    spacing_yx = None
    for path in files:
        ds = pydicom.dcmread(str(path), force=True)
        image = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        image = image * slope + intercept
        if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper() == "MONOCHROME1":
            image = np.max(image) - image
        slices.append(image)
        pos = getattr(ds, "ImagePositionPatient", None)
        if pos is not None and len(pos) >= 3:
            z_positions.append(float(pos[2]))
        elif hasattr(ds, "SliceLocation"):
            z_positions.append(float(ds.SliceLocation))
        elif hasattr(ds, "InstanceNumber"):
            z_positions.append(float(ds.InstanceNumber))
        spacing = getattr(ds, "PixelSpacing", None)
        if spacing is not None and len(spacing) >= 2:
            spacing_yx = (float(spacing[0]), float(spacing[1]))

    volume = np.stack(slices, axis=0).astype(np.float32)
    if len(z_positions) > 1:
        z_sorted = np.sort(np.asarray(z_positions, dtype=np.float32))
        dz = float(np.median(np.abs(np.diff(z_sorted)))) or float(getattr(ds, "SliceThickness", 1.0))
    else:
        dz = float(getattr(ds, "SliceThickness", 1.0))
    dy, dx = spacing_yx if spacing_yx is not None else (1.0, 1.0)
    return volume, (abs(dz), abs(dy), abs(dx))


def denoise_volume(volume_hu: np.ndarray) -> np.ndarray:
    """Lightweight slice-wise median denoising used before windowing/masking."""
    out = np.empty_like(volume_hu, dtype=np.float32)
    for z in range(volume_hu.shape[0]):
        # Work in int16-like HU values for numerical stability.
        out[z] = cv2.medianBlur(volume_hu[z].astype(np.float32), 3)
    return out


def resample_isotropic(volume: np.ndarray, spacing_zyx: Tuple[float, float, float], target_spacing: float) -> np.ndarray:
    """Resample [Z,H,W] volume to approximately isotropic spacing.

    This function uses scipy.ndimage.zoom if available. If scipy is absent, the
    original volume is returned so the training pipeline remains usable.
    """
    try:
        from scipy import ndimage as ndi
    except Exception:
        return volume
    z, y, x = spacing_zyx
    factors = (z / target_spacing, y / target_spacing, x / target_spacing)
    # Avoid extreme accidental resampling when DICOM spacing tags are malformed.
    factors = tuple(float(np.clip(f, 0.25, 4.0)) for f in factors)
    return ndi.zoom(volume, zoom=factors, order=1).astype(np.float32)


def segment_lung_mask(volume_hu: np.ndarray, cfg: DataConfig) -> np.ndarray:
    """Threshold/morphology lung mask used in the manuscript preprocessing.

    The default threshold isolates air-containing lung parenchyma and the
    optional morphology step removes small noisy regions and fills holes. This
    is intentionally simple and transparent; users can replace it with a more
    advanced lung segmentation mask when available.
    """
    mask = (volume_hu >= cfg.lung_mask_lower_hu) & (volume_hu <= cfg.lung_mask_upper_hu)
    if cfg.lung_mask_morphology:
        try:
            from scipy import ndimage as ndi

            mask = ndi.binary_opening(mask, iterations=1)
            mask = ndi.binary_closing(mask, iterations=2)
            mask = ndi.binary_fill_holes(mask)
            labeled, n = ndi.label(mask)
            if n > 0:
                sizes = ndi.sum(mask, labeled, index=np.arange(1, n + 1))
                keep = np.argsort(sizes)[-2:] + 1
                mask = np.isin(labeled, keep)
        except Exception:
            pass
    return mask.astype(np.uint8)


def window_normalize(volume_hu: np.ndarray, cfg: DataConfig) -> np.ndarray:
    lower = cfg.window_level - cfg.window_width / 2.0
    upper = cfg.window_level + cfg.window_width / 2.0
    volume = np.clip(volume_hu, lower, upper)
    volume = (volume - lower) / max(upper - lower, 1e-6)
    return volume.astype(np.float32)


def resize_slices(volume: np.ndarray, image_size: int) -> np.ndarray:
    out = np.empty((volume.shape[0], image_size, image_size), dtype=np.float32)
    for z in range(volume.shape[0]):
        out[z] = cv2.resize(volume[z], (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return out


def _cache_path(patient_dir: Path, cfg: DataConfig) -> Path | None:
    if cfg.preprocessed_cache_dir is None:
        return None
    key = f"{patient_dir.resolve()}|{cfg.image_size}|{cfg.window_width}|{cfg.window_level}|{cfg.use_lung_mask}|{cfg.enable_isotropic_resampling}|{cfg.target_spacing_mm}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    cfg.preprocessed_cache_dir.mkdir(parents=True, exist_ok=True)
    return cfg.preprocessed_cache_dir / f"{patient_dir.name}_{digest}.npz"


def preprocess_patient_volume(patient_dir: Path, cfg: DataConfig) -> Tuple[np.ndarray, np.ndarray | None]:
    """Full manuscript-aligned CT preprocessing pipeline.

    Steps: DICOM-to-HU conversion, optional denoising, optional isotropic 3D
    resampling, threshold/morphology lung-mask segmentation, lung-window
    clipping/normalization, optional lung-mask application, and spatial resizing.
    """
    cache = _cache_path(patient_dir, cfg)
    if cache is not None and cache.exists():
        data = np.load(cache)
        volume = data["volume"].astype(np.float32)
        mask = data["mask"].astype(np.uint8) if "mask" in data else None
        return volume, mask

    volume_hu, spacing = load_dicom_series(patient_dir)
    if cfg.use_denoising:
        volume_hu = denoise_volume(volume_hu)
    if cfg.enable_isotropic_resampling:
        volume_hu = resample_isotropic(volume_hu, spacing, cfg.target_spacing_mm)

    mask = None
    if cfg.use_lung_mask:
        mask = segment_lung_mask(volume_hu, cfg)

    volume = window_normalize(volume_hu, cfg)
    if cfg.use_lung_mask and cfg.apply_lung_mask_to_image and mask is not None:
        # Values outside lung are set to 0 after window normalization, matching a
        # masked lung-only image input while keeping image size unchanged.
        volume = volume * mask.astype(np.float32)

    volume = resize_slices(volume, cfg.image_size)
    if mask is not None:
        mask = resize_slices(mask.astype(np.float32), cfg.image_size) > 0.5
        mask = mask.astype(np.uint8)

    if cache is not None:
        np.savez_compressed(cache, volume=volume.astype(np.float32), mask=mask.astype(np.uint8) if mask is not None else np.zeros((0,), dtype=np.uint8))
    return volume.astype(np.float32), mask


def middle_slice_indices(num_slices: int, strip_ratio: float) -> List[int]:
    if num_slices <= 0:
        return []
    strip_ratio = min(max(float(strip_ratio), 0.0), 0.49)
    start = int(num_slices * strip_ratio)
    end = int(num_slices * (1.0 - strip_ratio))
    indices = list(range(start, end)) if end > start else list(range(num_slices))
    return indices or list(range(num_slices))

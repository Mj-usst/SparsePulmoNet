from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pydicom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_volume(patient_dir: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    files = [p for p in patient_dir.iterdir() if p.is_file()]
    slices = []
    spacings = []
    for p in files:
        ds = pydicom.dcmread(str(p), force=True)
        arr = ds.pixel_array.astype(np.float32)
        arr = arr * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
        pos = getattr(ds, "ImagePositionPatient", None)
        z = float(pos[2]) if pos is not None and len(pos) >= 3 else float(getattr(ds, "InstanceNumber", len(slices)))
        spacing_xy = getattr(ds, "PixelSpacing", [1.0, 1.0])
        spacings.append((float(spacing_xy[0]), float(spacing_xy[1]), z))
        slices.append((z, arr))
    if not slices:
        raise FileNotFoundError(f"No DICOM files found in {patient_dir}")
    slices = sorted(slices, key=lambda x: x[0])
    volume = np.stack([x[1] for x in slices], axis=0)
    if len(spacings) > 1:
        z_vals = sorted([s[2] for s in spacings])
        dz = float(np.median(np.diff(z_vals))) if len(z_vals) > 1 else 1.0
    else:
        dz = 1.0
    dy, dx = spacings[0][0], spacings[0][1]
    return volume, (abs(dz), dy, dx)


def _simple_lung_mask(volume_hu: np.ndarray) -> np.ndarray:
    # A lightweight default mask for open-source reproducibility. For publication-grade
    # radiomics, replace this with a validated lung segmentation mask when available.
    mask = (volume_hu > -1000) & (volume_hu < -300)
    try:
        from scipy import ndimage as ndi
        mask = ndi.binary_opening(mask, iterations=1)
        mask = ndi.binary_closing(mask, iterations=2)
    except Exception:
        pass
    return mask.astype(np.uint8)


def extract_one(patient_id: str, patient_dir: Path) -> Dict[str, float]:
    try:
        import SimpleITK as sitk
        from radiomics import featureextractor
    except ImportError as exc:
        raise ImportError("Radiomics extraction requires SimpleITK and pyradiomics. Install optional dependencies first.") from exc

    volume, spacing = _load_volume(patient_dir)
    mask = _simple_lung_mask(volume)
    image = sitk.GetImageFromArray(volume.astype(np.float32))
    mask_img = sitk.GetImageFromArray(mask.astype(np.uint8))
    image.SetSpacing(spacing[::-1])
    mask_img.SetSpacing(spacing[::-1])
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("shape")
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("glcm")
    extractor.enableFeatureClassByName("glrlm")
    extractor.enableFeatureClassByName("glszm")
    extractor.enableFeatureClassByName("ngtdm")
    result = extractor.execute(image, mask_img)
    row = {"Patient": patient_id}
    for key, value in result.items():
        if key.startswith("diagnostics"):
            continue
        try:
            row[str(key)] = float(value)
        except Exception:
            pass
    return row


def parse_args():
    p = argparse.ArgumentParser(description="Extract 3D radiomics features from OSIC DICOM folders using a simple lung-threshold mask")
    p.add_argument("--data-root", type=str, required=True, help="Directory containing train/<PatientID> DICOM folders")
    p.add_argument("--output", type=str, default="radiomics_all_features.csv")
    p.add_argument("--max-patients", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    train_dir = Path(args.data_root) / "train"
    patient_dirs = sorted([p for p in train_dir.iterdir() if p.is_dir()])
    if args.max_patients:
        patient_dirs = patient_dirs[: args.max_patients]
    rows: List[Dict[str, float]] = []
    for idx, patient_dir in enumerate(patient_dirs, start=1):
        print(f"[{idx}/{len(patient_dirs)}] {patient_dir.name}")
        try:
            rows.append(extract_one(patient_dir.name, patient_dir))
        except Exception as exc:
            print(f"  skipped: {exc}")
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

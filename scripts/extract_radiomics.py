from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sparsepulmonet.config import DataConfig
from sparsepulmonet.data.preprocessing import load_dicom_series, segment_lung_mask


def extract_one(patient_id: str, patient_dir: Path, cfg: DataConfig) -> Dict[str, float]:
    try:
        import SimpleITK as sitk
        from radiomics import featureextractor
    except ImportError as exc:
        raise ImportError("Radiomics extraction requires SimpleITK and pyradiomics. Install optional dependencies first.") from exc

    volume_hu, spacing_zyx = load_dicom_series(patient_dir)
    mask = segment_lung_mask(volume_hu, cfg)
    if mask.sum() == 0:
        raise RuntimeError("lung mask is empty")

    image = sitk.GetImageFromArray(volume_hu.astype(np.float32))
    mask_img = sitk.GetImageFromArray(mask.astype(np.uint8))
    # SimpleITK spacing order is x, y, z.
    image.SetSpacing(spacing_zyx[::-1])
    mask_img.SetSpacing(spacing_zyx[::-1])

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
    p = argparse.ArgumentParser(description="Extract 3D radiomics features from OSIC DICOM folders using the same threshold/morphology lung mask as model preprocessing")
    p.add_argument("--data-root", type=str, required=True, help="Directory containing train/<PatientID> DICOM folders")
    p.add_argument("--output", type=str, default="radiomics_all_features.csv")
    p.add_argument("--max-patients", type=int, default=None)
    p.add_argument("--lung-mask-lower-hu", type=float, default=-1000.0)
    p.add_argument("--lung-mask-upper-hu", type=float, default=-300.0)
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    cfg = DataConfig(
        data_root=data_root,
        radiomics_csv=Path(args.output),
        lung_mask_lower_hu=args.lung_mask_lower_hu,
        lung_mask_upper_hu=args.lung_mask_upper_hu,
    )
    train_dir = data_root / "train"
    patient_dirs = sorted([p for p in train_dir.iterdir() if p.is_dir()])
    if args.max_patients:
        patient_dirs = patient_dirs[: args.max_patients]
    rows: List[Dict[str, float]] = []
    for idx, patient_dir in enumerate(patient_dirs, start=1):
        print(f"[{idx}/{len(patient_dirs)}] {patient_dir.name}")
        try:
            rows.append(extract_one(patient_dir.name, patient_dir, cfg))
        except Exception as exc:
            print(f"  skipped: {exc}")
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

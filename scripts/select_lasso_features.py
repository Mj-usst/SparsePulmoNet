from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sparsepulmonet.config import DEFAULT_RADIOMICS_COLUMNS
from sparsepulmonet.data.osic import fit_patient_slope, read_osic_train_csv, read_radiomics_csv


def parse_args():
    p = argparse.ArgumentParser(description="Select radiomics features with LASSO for patient-level FVC slope prediction")
    p.add_argument("--train-csv", type=str, required=True)
    p.add_argument("--radiomics-csv", type=str, required=True)
    p.add_argument("--output", type=str, default="selected_radiomics.csv")
    p.add_argument("--feature-list", type=str, default="selected_radiomics_features.txt")
    p.add_argument("--use-manuscript-features", action="store_true", help="Subset the CSV to the nine manuscript features without refitting LASSO")
    return p.parse_args()


def main():
    args = parse_args()
    train_df = read_osic_train_csv(Path(args.train_csv))
    radiomics_df = read_radiomics_csv(Path(args.radiomics_csv))
    target = []
    patients = []
    for pid, pdf in train_df.groupby("Patient"):
        slope, _ = fit_patient_slope(pdf)
        patients.append(pid)
        target.append(slope)
    y_df = pd.DataFrame({"Patient": patients, "slope": target})
    merged = y_df.merge(radiomics_df, on="Patient", how="inner")

    if args.use_manuscript_features:
        selected = [c for c in DEFAULT_RADIOMICS_COLUMNS if c in merged.columns]
    else:
        non_feature = {"Patient", "slope"}
        feature_cols = [c for c in merged.columns if c not in non_feature and pd.api.types.is_numeric_dtype(merged[c])]
        x = merged[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        x = StandardScaler().fit_transform(x)
        y = merged["slope"].to_numpy(dtype=np.float32)
        model = LassoCV(cv=5, random_state=42, max_iter=10000).fit(x, y)
        selected = [col for col, coef in zip(feature_cols, model.coef_) if abs(coef) > 1e-8]
        if not selected:
            print("LASSO selected no nonzero features; falling back to manuscript feature list where available.")
            selected = [c for c in DEFAULT_RADIOMICS_COLUMNS if c in merged.columns]

    out = radiomics_df[["Patient", *selected]].copy()
    out.to_csv(args.output, index=False)
    Path(args.feature_list).write_text("\n".join(selected), encoding="utf-8")
    print(f"Selected {len(selected)} features")
    print(f"Saved {args.output} and {args.feature_list}")


if __name__ == "__main__":
    main()

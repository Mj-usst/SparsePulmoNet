from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def parse_args():
    p = argparse.ArgumentParser(description="Run Table 5 fusion and backbone ablations")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--radiomics-csv", type=str, required=True)
    p.add_argument("--save-root", type=str, default="outputs/table5_fusion_backbone_ablation")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--fusion-types", type=str, default="concat,attention,vae")
    p.add_argument("--backbones", type=str, default="resnet50,resnext50_32x4d,efficientnet_b0,crate_tiny")
    p.add_argument("--extra-args", type=str, default="", help="Extra arguments passed verbatim to scripts/train_osic.py")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _load_summary(path: Path) -> Dict[str, float]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not rows:
        return {"lllm": float("nan"), "rmse": float("nan"), "mae": float("nan")}
    return {metric: sum(float(r[metric]) for r in rows) / len(rows) for metric in ["lllm", "rmse", "mae"]}


def main():
    args = parse_args()
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    train_script = Path(__file__).resolve().parent / "train_osic.py"

    base = [
        sys.executable,
        str(train_script),
        "--data-root", args.data_root,
        "--radiomics-csv", args.radiomics_csv,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--folds", str(args.folds),
        "--num-workers", str(args.num_workers),
    ]
    if args.extra_args:
        base += args.extra_args.split()

    summary_rows = []

    # Fusion ablation with the manuscript CRATE backbone.
    for fusion_type in [x.strip() for x in args.fusion_types.split(",") if x.strip()]:
        name = f"crate_tiny_{fusion_type}"
        out_dir = save_root / "fusion" / name
        cmd = base + ["--model-name", "crate_tiny", "--fusion-type", fusion_type, "--save-dir", str(out_dir)]
        print("\n" + " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
            metrics = _load_summary(out_dir / "cv_summary.json")
            summary_rows.append({"ablation_group": "fusion", "experiment": name, "model_name": "crate_tiny", "fusion_type": fusion_type, **metrics})

    # Backbone ablation with VAE fusion.
    for backbone in [x.strip() for x in args.backbones.split(",") if x.strip()]:
        name = f"{backbone}_vae"
        out_dir = save_root / "backbone" / name
        cmd = base + ["--model-name", backbone, "--fusion-type", "vae", "--save-dir", str(out_dir)]
        print("\n" + " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
            metrics = _load_summary(out_dir / "cv_summary.json")
            summary_rows.append({"ablation_group": "backbone", "experiment": name, "model_name": backbone, "fusion_type": "vae", **metrics})

    if summary_rows:
        out_csv = save_root / "table5_fusion_backbone_ablation_summary.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()

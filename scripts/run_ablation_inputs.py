from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def parse_args():
    p = argparse.ArgumentParser(description="Run Table 3 input-combination ablations: image-only, image+clinical, image+clinical+radiomics")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--radiomics-csv", type=str, required=True)
    p.add_argument("--save-root", type=str, default="outputs/table3_input_ablation")
    p.add_argument("--model-name", type=str, default="crate_tiny")
    p.add_argument("--fusion-type", type=str, default="vae", choices=["vae", "concat", "attention"])
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--extra-args", type=str, default="", help="Extra arguments passed verbatim to scripts/train_osic.py")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _cmd(base: List[str], flags: List[str]) -> List[str]:
    return base + flags


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
        "--model-name", args.model_name,
        "--fusion-type", args.fusion_type,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--folds", str(args.folds),
        "--num-workers", str(args.num_workers),
    ]
    if args.extra_args:
        base += args.extra_args.split()

    experiments = [
        ("image_only", ["--fusion-type", "image_only", "--no-clinical", "--no-radiomics"]),
        ("image_clinical", ["--no-radiomics"]),
        ("image_clinical_radiomics", []),
    ]

    summary_rows = []
    for name, flags in experiments:
        out_dir = save_root / name
        cmd = _cmd(base + ["--save-dir", str(out_dir)], flags)
        print("\n" + " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
            metrics = _load_summary(out_dir / "cv_summary.json")
            summary_rows.append({"experiment": name, **metrics})

    if summary_rows:
        out_csv = save_root / "table3_input_ablation_summary.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()

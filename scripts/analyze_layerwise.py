from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sparsepulmonet.config import DataConfig, ModelConfig
from sparsepulmonet.data.osic import make_osic_folds
from sparsepulmonet.models.ipf_model import SparsePulmoNet
from sparsepulmonet.utils.layerwise import summarize_layerwise
from sparsepulmonet.utils.seed import seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="Compute CRATE layer-wise coding-rate and sparsity proxies")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--radiomics-csv", type=str, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--output-csv", type=str, default="outputs/layerwise_metrics.csv")
    p.add_argument("--output-png", type=str, default="outputs/layerwise_metrics.png")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--model-name", type=str, default="crate_tiny", choices=["crate_tiny", "crate_small", "crate_base", "crate_large"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    data_cfg = DataConfig(
        data_root=Path(args.data_root),
        radiomics_csv=Path(args.radiomics_csv),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        strict=True,
        random_state=args.seed,
    )
    model_cfg = ModelConfig(model_name=args.model_name, image_size=args.image_size)
    folds = make_osic_folds(data_cfg)
    fold_info = folds[args.fold]
    loader = fold_info["val_loader"] if args.split == "val" else fold_info["train_loader"]
    model = SparsePulmoNet(model_cfg, tabular_dim=data_cfg.tabular_feature_dim).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    rows_accum = []
    with torch.no_grad():
        for images, tabular, _targets, _metas in loader:
            images = images.to(device)
            tabular = tabular.to(device)
            _pred, aux = model(images, tabular, return_layerwise=True)
            rows = summarize_layerwise(aux["layerwise"])
            rows_accum.append(rows)
            break

    if not rows_accum:
        raise RuntimeError("No batches were available for layer-wise analysis")
    rows = rows_accum[0]
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt
        layers = [r["layer"] for r in rows]
        coding = [r["coding_rate_mean"] for r in rows]
        zeros = [r["zero_fraction"] for r in rows]
        fig = plt.figure(figsize=(7, 4))
        ax1 = fig.add_subplot(111)
        ax1.plot(layers, coding, marker="o", label="coding-rate proxy")
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Coding-rate proxy")
        ax2 = ax1.twinx()
        ax2.plot(layers, zeros, marker="s", label="zero fraction")
        ax2.set_ylabel("Zero fraction")
        fig.tight_layout()
        out_png = Path(args.output_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=200)
    except Exception as exc:
        print(f"Plot generation skipped: {exc}")

    print(f"Saved layer-wise metrics to {out_csv}")


if __name__ == "__main__":
    main()

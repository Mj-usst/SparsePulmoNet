from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sparsepulmonet.config import DataConfig, ModelConfig
from sparsepulmonet.data.osic import make_osic_folds
from sparsepulmonet.models.ipf_model import SparsePulmoNet
from sparsepulmonet.training.engine import evaluate
from sparsepulmonet.utils.seed import seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a saved SparsePulmoNet checkpoint on one OSIC CV fold")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--radiomics-csv", type=str, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--model-name", type=str, default="crate_tiny", choices=["crate_tiny", "crate_small", "crate_base", "crate_large", "resnet18", "resnet50", "resnext50_32x4d", "efficientnet_b0"])
    p.add_argument("--fusion-type", type=str, default="vae", choices=["vae", "concat", "attention", "image_only"])
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--fixed-sigma", type=float, default=70.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-clinical", action="store_true")
    p.add_argument("--no-radiomics", action="store_true")
    p.add_argument("--disable-denoising", action="store_true")
    p.add_argument("--disable-lung-mask", action="store_true")
    p.add_argument("--disable-isotropic-resampling", action="store_true")
    p.add_argument("--do-not-apply-lung-mask-to-image", action="store_true")
    p.add_argument("--target-spacing-mm", type=float, default=1.0)
    p.add_argument("--preprocessed-cache-dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    use_clinical = not args.no_clinical
    use_radiomics = not args.no_radiomics
    if args.fusion_type == "image_only":
        use_clinical = False
        use_radiomics = False
    data_cfg = DataConfig(
        data_root=Path(args.data_root),
        radiomics_csv=Path(args.radiomics_csv),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_state=args.seed,
        use_clinical=use_clinical,
        use_radiomics=use_radiomics,
        use_denoising=not args.disable_denoising,
        use_lung_mask=not args.disable_lung_mask,
        apply_lung_mask_to_image=not args.do_not_apply_lung_mask_to_image,
        enable_isotropic_resampling=not args.disable_isotropic_resampling,
        target_spacing_mm=args.target_spacing_mm,
        preprocessed_cache_dir=Path(args.preprocessed_cache_dir) if args.preprocessed_cache_dir else None,
    )
    model_cfg = ModelConfig(model_name=args.model_name, fusion_type=args.fusion_type, image_size=args.image_size)
    folds = make_osic_folds(data_cfg)
    fold_info = folds[args.fold]
    model = SparsePulmoNet(model_cfg, tabular_dim=data_cfg.tabular_feature_dim).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=True)
    metrics = evaluate(
        model,
        fold_info["val_loader"],
        fold_info["val_patients"],
        fold_info["train_df"],
        device,
        use_amp=False,
        fixed_sigma=args.fixed_sigma,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

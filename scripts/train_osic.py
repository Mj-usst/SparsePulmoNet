from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sparsepulmonet.config import DataConfig, ModelConfig, TrainConfig
from sparsepulmonet.data.osic import make_osic_folds, write_fold_summary
from sparsepulmonet.models.ipf_model import SparsePulmoNet
from sparsepulmonet.training.engine import evaluate, train_one_epoch
from sparsepulmonet.utils.seed import seed_everything


def _parse_patient_ids(value: str | None) -> List[str]:
    if not value:
        return []
    path = Path(value)
    if path.exists():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Train SparsePulmoNet on the OSIC Pulmonary Fibrosis Progression dataset")
    parser.add_argument("--data-root", type=str, required=True, help="Directory containing train.csv and train/<PatientID> DICOM folders")
    parser.add_argument("--radiomics-csv", type=str, required=True, help="CSV containing one row per patient and selected radiomics columns")
    parser.add_argument("--save-dir", type=str, default="outputs/sparsepulmonet_osic")

    parser.add_argument("--model-name", type=str, default="crate_tiny", choices=["crate_tiny", "crate_small", "crate_base", "crate_large", "resnet18", "resnet50", "resnext50_32x4d", "efficientnet_b0"])
    parser.add_argument("--fusion-type", type=str, default="vae", choices=["vae", "concat", "attention", "image_only"])
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--pool", type=str, default="cls", choices=["cls", "mean"])
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)

    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--fixed-sigma", type=float, default=70.0, help="Fixed FVC uncertainty used for LLLm when the model predicts point estimates only")
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=1e-4)

    # Table 3 input ablation switches.
    parser.add_argument("--no-clinical", action="store_true", help="Disable age/sex/smoking clinical inputs")
    parser.add_argument("--no-radiomics", action="store_true", help="Disable radiomics inputs")

    # Manuscript preprocessing switches.
    parser.add_argument("--disable-denoising", action="store_true")
    parser.add_argument("--disable-lung-mask", action="store_true")
    parser.add_argument("--disable-isotropic-resampling", action="store_true")
    parser.add_argument("--do-not-apply-lung-mask-to-image", action="store_true")
    parser.add_argument("--target-spacing-mm", type=float, default=1.0)
    parser.add_argument("--preprocessed-cache-dir", type=str, default=None)

    parser.add_argument("--exclude-patient-ids", type=str, default=None, help="Comma-separated IDs or a text file with one patient ID per line")
    parser.add_argument("--non-strict", action="store_true", help="Skip patients with missing DICOM/radiomics instead of raising an error")
    parser.add_argument("--debug-max-folds", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.preprocessed_cache_dir) if args.preprocessed_cache_dir else save_dir / "preprocessed_cache"
    use_clinical = not args.no_clinical
    use_radiomics = not args.no_radiomics
    if args.fusion_type == "image_only":
        use_clinical = False
        use_radiomics = False

    data_cfg = DataConfig(
        data_root=Path(args.data_root),
        radiomics_csv=Path(args.radiomics_csv),
        image_size=args.image_size,
        folds=args.folds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        exclude_patient_ids=_parse_patient_ids(args.exclude_patient_ids),
        strict=not args.non_strict,
        random_state=args.seed,
        use_clinical=use_clinical,
        use_radiomics=use_radiomics,
        use_denoising=not args.disable_denoising,
        use_lung_mask=not args.disable_lung_mask,
        apply_lung_mask_to_image=not args.do_not_apply_lung_mask_to_image,
        enable_isotropic_resampling=not args.disable_isotropic_resampling,
        target_spacing_mm=args.target_spacing_mm,
        preprocessed_cache_dir=cache_dir,
    )
    model_cfg = ModelConfig(
        model_name=args.model_name,
        fusion_type=args.fusion_type,
        image_size=args.image_size,
        patch_size=args.patch_size,
        latent_dim=args.latent_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        pool=args.pool,
        predict_sigma=False,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        use_amp=not args.no_amp,
        recon_weight=args.recon_weight,
        kl_weight=args.kl_weight,
        fixed_sigma=args.fixed_sigma,
        save_dir=save_dir,
    )

    device = torch.device(train_cfg.device if torch.cuda.is_available() and train_cfg.device != "cpu" else "cpu")
    folds = make_osic_folds(data_cfg)
    if args.debug_max_folds:
        folds = folds[: args.debug_max_folds]
    write_fold_summary(folds, save_dir / "fold_summary.json")

    config_json = {
        "data_cfg": {**asdict(data_cfg), "data_root": str(data_cfg.data_root), "radiomics_csv": str(data_cfg.radiomics_csv), "preprocessed_cache_dir": str(data_cfg.preprocessed_cache_dir)},
        "model_cfg": asdict(model_cfg),
        "train_cfg": {**asdict(train_cfg), "save_dir": str(train_cfg.save_dir), "resume": str(train_cfg.resume) if train_cfg.resume else None},
    }
    (save_dir / "run_config.json").write_text(json.dumps(config_json, indent=2, ensure_ascii=False), encoding="utf-8")

    all_results = []
    for fold_info in folds:
        fold = int(fold_info["fold"])
        print(f"\n===== Fold {fold + 1}/{len(folds)} | train={len(fold_info['train_loader'].dataset)} val={len(fold_info['val_loader'].dataset)} =====")
        model = SparsePulmoNet(model_cfg, tabular_dim=data_cfg.tabular_feature_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.use_amp and device.type == "cuda")

        best_lllm = float("-inf")
        best_metrics = None
        run_name = f"{model_cfg.model_name}_{model_cfg.fusion_type}_fold{fold}"
        ckpt_path = save_dir / f"{run_name}.pt"

        for epoch in range(train_cfg.epochs):
            train_metrics = train_one_epoch(
                model=model,
                loader=fold_info["train_loader"],
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                use_amp=train_cfg.use_amp and device.type == "cuda",
                slope_loss_weight=train_cfg.slope_loss_weight,
                recon_weight=train_cfg.recon_weight,
                kl_weight=train_cfg.kl_weight,
                desc=f"fold{fold} epoch{epoch+1}/{train_cfg.epochs}",
            )
            val_metrics = evaluate(
                model=model,
                loader=fold_info["val_loader"],
                val_patients=fold_info["val_patients"],
                train_df=fold_info["train_df"],
                device=device,
                use_amp=train_cfg.use_amp and device.type == "cuda",
                fixed_sigma=train_cfg.fixed_sigma,
                desc=f"fold{fold} val",
            )
            print(
                f"epoch={epoch + 1:03d} "
                f"loss={train_metrics['loss']:.4f} "
                f"val_LLLm={val_metrics['lllm']:.4f} "
                f"val_RMSE={val_metrics['rmse']:.2f} "
                f"val_MAE={val_metrics['mae']:.2f}"
            )
            if val_metrics["lllm"] > best_lllm:
                best_lllm = val_metrics["lllm"]
                best_metrics = {"epoch": epoch + 1, **train_metrics, **val_metrics}
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "model_cfg": asdict(model_cfg),
                        "data_cfg": {
                            "image_size": data_cfg.image_size,
                            "tabular_feature_dim": data_cfg.tabular_feature_dim,
                            "use_clinical": data_cfg.use_clinical,
                            "use_radiomics": data_cfg.use_radiomics,
                        },
                        "metrics": best_metrics,
                    },
                    ckpt_path,
                )
        if best_metrics is not None:
            all_results.append({"fold": fold, "model_name": model_cfg.model_name, "fusion_type": model_cfg.fusion_type, "use_clinical": data_cfg.use_clinical, "use_radiomics": data_cfg.use_radiomics, **best_metrics})
            print(f"Best fold {fold}: {best_metrics}")

    summary_path = save_dir / "cv_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    if all_results:
        mean = {key: sum(item[key] for item in all_results) / len(all_results) for key in ["lllm", "rmse", "mae"]}
        print(f"\nCV mean: LLLm={mean['lllm']:.4f}, RMSE={mean['rmse']:.2f}, MAE={mean['mae']:.2f}")


if __name__ == "__main__":
    main()

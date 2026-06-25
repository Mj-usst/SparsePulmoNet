# SparsePulmoNet

SparsePulmoNet is a CRATE-based multimodal framework for CT-based prediction of idiopathic pulmonary fibrosis (IPF) progression on the public OSIC Pulmonary Fibrosis Progression dataset.

The code follows the manuscript implementation:

- **Dataset**: OSIC `train.csv` + baseline chest CT DICOM folders.
- **Patient-level target**: ordinary-least-squares FVC slope from longitudinal FVC records.
- **Preprocessing**: DICOM-to-HU conversion, denoising, isotropic 3D resampling, lung-window clipping, threshold/morphology lung-mask segmentation, masked lung input, and spatial resizing.
- **Image branch**: CRATE/Coding RAte reduction TransformEr with MSSA-style attention and ISTA-style sparse-coding blocks.
- **Tabular branch**: age, sex, smoking status, and selected radiomics features.
- **Fusion**: VAE-style multimodal fusion with reconstruction and KL regularization.
- **Evaluation**: modified Laplace log likelihood (LLLm), RMSE, and MAE at the visit level.

> The OSIC dataset is not included. Download it from Kaggle and place it locally according to the structure below.

## Repository structure

```text
SparsePulmoNet/
├── src/sparsepulmonet/
│   ├── config.py
│   ├── data/
│   │   ├── osic.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── baseline_backbones.py
│   │   ├── crate_backbone.py
│   │   ├── fusion.py
│   │   └── ipf_model.py
│   ├── training/
│   │   ├── engine.py
│   │   └── metrics.py
│   └── utils/layerwise.py
├── scripts/
│   ├── train_osic.py
│   ├── evaluate_checkpoint.py
│   ├── analyze_layerwise.py
│   ├── extract_radiomics.py
│   ├── select_lasso_features.py
│   ├── run_ablation_inputs.py
│   └── run_ablation_fusion_backbones.py
├── tests/test_forward.py
├── requirements.txt
├── pyproject.toml
├── NOTICE
└── LICENSE
```

## Data layout

Expected OSIC layout:

```text
/path/to/osic/
├── train.csv
└── train/
    ├── ID00007637202177411956430/
    │   ├── 1.dcm
    │   ├── 2.dcm
    │   └── ...
    └── ...
```

Expected selected radiomics CSV:

```text
Patient,original_shape_LeastAxisLength,original_shape_Maximum2DDiameterSlice,...
ID00007637202177411956430, ...
```

The default selected feature names match the manuscript:

1. `original_shape_LeastAxisLength`
2. `original_shape_Maximum2DDiameterSlice`
3. `original_shape_SurfaceVolumeRatio`
4. `original_firstorder_10Percentile`
5. `original_firstorder_Skewness`
6. `original_glcm_ClusterShade`
7. `original_glrlm_LongRunHighGrayLevelEmphasis`
8. `original_glszm_HighGrayLevelZoneEmphasis`
9. `original_ngtdm_Strength`

## Installation

```bash
git clone https://github.com/Mj-usst/SparsePulmoNet.git
cd SparsePulmoNet
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Optional radiomics extraction requires:

```bash
pip install pyradiomics SimpleITK
```

## Main model training

The default training command uses the manuscript-style settings: five-fold CV, batch size 4, AdamW, learning rate `5e-5`, and 400 epochs. By default, preprocessing includes denoising, isotropic resampling, and lung-mask segmentation.

```bash
python scripts/train_osic.py \
  --data-root /path/to/osic \
  --radiomics-csv /path/to/selected_radiomics.csv \
  --model-name crate_tiny \
  --fusion-type vae \
  --image-size 512 \
  --batch-size 4 \
  --epochs 400 \
  --lr 5e-5 \
  --save-dir outputs/sparsepulmonet_osic
```

Outputs:

```text
outputs/sparsepulmonet_osic/
├── run_config.json
├── fold_summary.json
├── crate_tiny_vae_fold0.pt
├── crate_tiny_vae_fold1.pt
├── ...
├── preprocessed_cache/
└── cv_summary.json
```

## Manuscript preprocessing

The model input preprocessing is implemented in `src/sparsepulmonet/data/preprocessing.py` and includes:

1. DICOM series sorting by spatial position or instance number.
2. Conversion to HU with `RescaleSlope` and `RescaleIntercept`.
3. Slice-wise median denoising.
4. Optional 3D isotropic resampling to 1.0 mm spacing.
5. Threshold/morphology lung-mask segmentation.
6. Lung-window clipping with WW/WL = 1600/-600 HU and normalization to `[0, 1]`.
7. Optional lung-mask application to suppress non-lung regions.
8. Spatial resizing to `image_size × image_size`.
9. Middle-lung slice selection after excluding the apical and basal 30% of slices.

Useful switches:

```bash
--disable-denoising
--disable-lung-mask
--disable-isotropic-resampling
--do-not-apply-lung-mask-to-image
--target-spacing-mm 1.0
--preprocessed-cache-dir outputs/preprocessed_cache
```

## Table 3: input-combination ablation

This script runs the three input settings reported in Table 3:

- image-only
- image + clinical
- image + clinical + radiomics

```bash
python scripts/run_ablation_inputs.py \
  --data-root /path/to/osic \
  --radiomics-csv /path/to/selected_radiomics.csv \
  --save-root outputs/table3_input_ablation \
  --model-name crate_tiny \
  --fusion-type vae \
  --epochs 400 \
  --batch-size 4 \
  --folds 5
```

The script writes:

```text
outputs/table3_input_ablation/table3_input_ablation_summary.csv
```

## Table 5: fusion and backbone ablation

This script runs fusion ablations and backbone ablations:

- fusion: concat, attention, VAE
- backbone: ResNet, ResNeXt, EfficientNet, CRATE

```bash
python scripts/run_ablation_fusion_backbones.py \
  --data-root /path/to/osic \
  --radiomics-csv /path/to/selected_radiomics.csv \
  --save-root outputs/table5_fusion_backbone_ablation \
  --epochs 400 \
  --batch-size 4 \
  --folds 5
```

The script writes:

```text
outputs/table5_fusion_backbone_ablation/table5_fusion_backbone_ablation_summary.csv
```

## Evaluate a checkpoint

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint outputs/sparsepulmonet_osic/crate_tiny_vae_fold0.pt \
  --data-root /path/to/osic \
  --radiomics-csv /path/to/selected_radiomics.csv \
  --fold 0
```

## Layer-wise interpretability analysis

This script computes the coding-rate proxy at the post-MSSA stage and the zero/nonzero fraction at the post-ISTA stage.

```bash
python scripts/analyze_layerwise.py \
  --checkpoint outputs/sparsepulmonet_osic/crate_tiny_vae_fold0.pt \
  --data-root /path/to/osic \
  --radiomics-csv /path/to/selected_radiomics.csv \
  --fold 0 \
  --output-csv outputs/layerwise_metrics_fold0.csv \
  --output-png outputs/layerwise_metrics_fold0.png
```

## Radiomics workflow

To extract radiomics features from DICOM folders using a threshold/morphology lung mask:

```bash
python scripts/extract_radiomics.py \
  --data-root /path/to/osic \
  --output radiomics_all_features.csv
```

Then select features using LASSO:

```bash
python scripts/select_lasso_features.py \
  --train-csv /path/to/osic/train.csv \
  --radiomics-csv radiomics_all_features.csv \
  --output selected_radiomics.csv
```

For exact manuscript-feature subsetting without refitting LASSO:

```bash
python scripts/select_lasso_features.py \
  --train-csv /path/to/osic/train.csv \
  --radiomics-csv radiomics_all_features.csv \
  --use-manuscript-features \
  --output selected_radiomics.csv
```

## Important reproducibility notes

1. No patients are excluded by default. If corrupted or incomplete cases must be excluded, pass `--exclude-patient-ids` and report the final patient count in the manuscript.
2. DICOM pixels are converted to HU using `RescaleSlope` and `RescaleIntercept`, denoised, optionally isotropically resampled, segmented by a lung mask, clipped to WW/WL = 1600/-600 HU, and normalized to `[0, 1]`.
3. Training samples use a random slice from the middle 40% of each CT scan; validation samples use the center slice from that same middle region.
4. The model predicts a patient-specific FVC slope. Visit-level FVC is reconstructed from the observed baseline FVC closest to week 0.
5. Because this implementation predicts point estimates, LLLm uses a fixed uncertainty of 70 mL by default. If the manuscript uses another uncertainty definition, this must be reported and implemented explicitly.
6. The VAE fusion module is trained with both reconstruction and KL terms by default (`--recon-weight 1.0`, `--kl-weight 1e-4`). These losses are automatically disabled for concat/attention fusion ablations.

## Citation

Please cite both the SparsePulmoNet manuscript and the original CRATE paper:

```text
Yu Y, Buchanan S, Pai D, et al. White-Box Transformers via Sparse Rate Reduction. NeurIPS 2023.
```

## License

This repository is released under the MIT License. The adapted CRATE component is based on MIT-licensed CRATE code; see `third_party/CRATE_LICENSE`.

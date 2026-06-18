# SparsePulmoNet

SparsePulmoNet is a CRATE-based multimodal framework for CT-based prediction of idiopathic pulmonary fibrosis (IPF) progression on the public OSIC Pulmonary Fibrosis Progression dataset.

The code follows the manuscript implementation as closely as possible:

- **Dataset**: OSIC `train.csv` + baseline chest CT DICOM folders.
- **Patient-level target**: ordinary-least-squares FVC slope from longitudinal FVC records.
- **Image branch**: CRATE/Coding RAte reduction TransformEr with MSSA-style attention and ISTA-style sparse-coding blocks.
- **Tabular branch**: age, sex, smoking status, and selected radiomics features.
- **Fusion**: VAE-style multimodal fusion with reconstruction and KL regularization.
- **Evaluation**: modified Laplace log likelihood (LLLm), RMSE, and MAE at the visit level.

> The OSIC dataset is not included. Download it from Kaggle and place it locally according to the structure below.

## Repository structure

```text
SparsePulmoNet_OpenSource/
├── src/sparsepulmonet/
│   ├── config.py
│   ├── data/osic.py
│   ├── models/crate_backbone.py
│   ├── models/fusion.py
│   ├── models/ipf_model.py
│   ├── training/engine.py
│   ├── training/metrics.py
│   └── utils/layerwise.py
├── scripts/
│   ├── train_osic.py
│   ├── evaluate_checkpoint.py
│   ├── analyze_layerwise.py
│   ├── extract_radiomics.py
│   └── select_lasso_features.py
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
cd SparsePulmoNet_OpenSource
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Optional radiomics extraction requires:

```bash
pip install pyradiomics SimpleITK
```

## Train five-fold OSIC model

The default training command uses the manuscript-style settings: five-fold CV, batch size 4, AdamW, learning rate `5e-5`, and 400 epochs.

```bash
python scripts/train_osic.py \
  --data-root /path/to/osic \
  --radiomics-csv /path/to/selected_radiomics.csv \
  --model-name crate_tiny \
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
├── crate_tiny_fold0.pt
├── crate_tiny_fold1.pt
├── ...
└── cv_summary.json
```

## Evaluate a checkpoint

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint outputs/sparsepulmonet_osic/crate_tiny_fold0.pt \
  --data-root /path/to/osic \
  --radiomics-csv /path/to/selected_radiomics.csv \
  --fold 0
```

## Layer-wise interpretability analysis

This script computes the coding-rate proxy at the post-MSSA stage and the zero/nonzero fraction at the post-ISTA stage.

```bash
python scripts/analyze_layerwise.py \
  --checkpoint outputs/sparsepulmonet_osic/crate_tiny_fold0.pt \
  --data-root /path/to/osic \
  --radiomics-csv /path/to/selected_radiomics.csv \
  --fold 0 \
  --output-csv outputs/layerwise_metrics_fold0.csv \
  --output-png outputs/layerwise_metrics_fold0.png
```

## Radiomics workflow

To extract radiomics features from DICOM folders using a lightweight threshold-based lung mask:

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
2. DICOM pixels are converted to HU using `RescaleSlope` and `RescaleIntercept`, then clipped to WW/WL = 1600/-600 HU and normalized to `[0, 1]`.
3. Training samples use a random slice from the middle 40% of each CT scan; validation samples use the center slice from that same middle region.
4. The model predicts a patient-specific FVC slope. Visit-level FVC is reconstructed from the observed baseline FVC closest to week 0.
5. Because this implementation predicts point estimates, LLLm uses a fixed uncertainty of 70 mL by default. If the manuscript uses another uncertainty definition, this must be reported and implemented explicitly.
6. The VAE fusion module is trained with both reconstruction and KL terms by default (`--recon-weight 1.0`, `--kl-weight 1e-4`).

## Citation

Please cite both the SparsePulmoNet manuscript and the original CRATE paper:

```text
Yu Y, Buchanan S, Pai D, et al. White-Box Transformers via Sparse Rate Reduction. NeurIPS 2023.
```

## License

This repository is released under the MIT License. The adapted CRATE component is based on MIT-licensed CRATE code; see `third_party/CRATE_LICENSE`.

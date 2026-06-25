from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

DEFAULT_RADIOMICS_COLUMNS = [
    "original_shape_LeastAxisLength",
    "original_shape_Maximum2DDiameterSlice",
    "original_shape_SurfaceVolumeRatio",
    "original_firstorder_10Percentile",
    "original_firstorder_Skewness",
    "original_glcm_ClusterShade",
    "original_glrlm_LongRunHighGrayLevelEmphasis",
    "original_glszm_HighGrayLevelZoneEmphasis",
    "original_ngtdm_Strength",
]


@dataclass
class DataConfig:
    """Dataset and preprocessing configuration.

    The defaults reflect the manuscript workflow: OSIC train.csv, baseline CT
    DICOM folders, HU conversion, lung-window preprocessing, denoising,
    threshold/morphology lung-mask segmentation, optional isotropic 3D
    resampling before middle-lung slice selection, 5-fold cross-validation,
    and no default patient exclusion.
    """

    data_root: Path
    radiomics_csv: Path
    train_csv_name: str = "train.csv"
    train_dir_name: str = "train"
    image_size: int = 512
    window_width: float = 1600.0
    window_level: float = -600.0
    strip_ratio: float = 0.30
    folds: int = 5
    num_workers: int = 4
    batch_size: int = 4
    radiomics_columns: List[str] = field(default_factory=lambda: list(DEFAULT_RADIOMICS_COLUMNS))
    exclude_patient_ids: List[str] = field(default_factory=list)
    strict: bool = True
    train_slice_policy: Literal["random_middle", "center_middle"] = "random_middle"
    eval_slice_policy: Literal["center_middle", "random_middle"] = "center_middle"
    normalize_radiomics_per_fold: bool = True
    use_clinical: bool = True
    use_radiomics: bool = True

    # CT preprocessing used before 2D middle-lung slice extraction.
    use_denoising: bool = True
    use_lung_mask: bool = True
    lung_mask_lower_hu: float = -1000.0
    lung_mask_upper_hu: float = -300.0
    lung_mask_morphology: bool = True
    apply_lung_mask_to_image: bool = True
    enable_isotropic_resampling: bool = True
    target_spacing_mm: float = 1.0
    preprocessed_cache_dir: Optional[Path] = None
    random_state: int = 42

    @property
    def train_csv(self) -> Path:
        return self.data_root / self.train_csv_name

    @property
    def train_dir(self) -> Path:
        return self.data_root / self.train_dir_name

    @property
    def clinical_feature_dim(self) -> int:
        return 4 if self.use_clinical else 0  # age + sex + two-bit smoking status

    @property
    def radiomics_feature_dim(self) -> int:
        return len(self.radiomics_columns) if self.use_radiomics else 0

    @property
    def tabular_feature_dim(self) -> int:
        return self.clinical_feature_dim + self.radiomics_feature_dim


@dataclass
class ModelConfig:
    model_name: str = "crate_tiny"  # crate_tiny/small/base/large, resnet18/50, resnext50_32x4d, efficientnet_b0
    fusion_type: Literal["vae", "concat", "attention", "image_only"] = "vae"
    image_size: int = 512
    patch_size: int = 16
    in_channels: int = 1
    latent_dim: int = 32
    fusion_hidden_dim: int = 256
    ista_step_size: float = 0.1
    ista_lambda: float = 0.1
    dropout: float = 0.0
    emb_dropout: float = 0.0
    pool: Literal["cls", "mean"] = "cls"
    predict_sigma: bool = False


@dataclass
class TrainConfig:
    epochs: int = 400
    lr: float = 5e-5
    weight_decay: float = 1e-2
    seed: int = 42
    device: str = "cuda"
    use_amp: bool = True
    slope_loss_weight: float = 1.0
    recon_weight: float = 1.0
    kl_weight: float = 1e-4
    sigma_loss_weight: float = 0.0
    fixed_sigma: float = 70.0
    save_dir: Path = Path("outputs")
    resume: Optional[Path] = None

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RuntimeConfig:
    device: str = "cuda:0"
    verbose: bool = True


@dataclass
class DatasetConfig:
    is_train: bool = False
    datasets_to_process: list[str] | None = None
    max_images: int | None = None


@dataclass
class PathsConfig:
    data_dir: str = "/kaggle/input/image-matching-challenge-2025"
    work_dir: str = "/kaggle/working"
    dino_model_path: str = "/kaggle/input/dinov2/pytorch/base/1"
    metric_module_dir: str = "/kaggle/input/imc25-utils"
    train_csv: str = "/kaggle/input/image-matching-challenge-2025/train_labels.csv"
    sample_submission_csv: str = "/kaggle/input/image-matching-challenge-2025/sample_submission.csv"
    thresholds_csv: str = "/kaggle/input/image-matching-challenge-2025/train_thresholds.csv"


@dataclass
class PairingConfig:
    similarity_threshold: float = 0.3
    min_pairs_per_image: int = 20
    exhaustive_if_less: int = 20


@dataclass
class FeaturesConfig:
    max_keypoints: int = 4096
    resize_to: int = 1024
    detection_threshold: float = 0.01
    use_rotation_augmentation: bool = True
    rotations: list[int] | None = None


@dataclass
class MatchingConfig:
    min_matches: int = 25
    width_confidence: float = -1
    depth_confidence: float = -1


@dataclass
class ReconstructionConfig:
    min_model_size: int = 3
    max_num_models: int = 25


@dataclass
class ExperimentConfig:
    experiment_name: str
    runtime: RuntimeConfig
    dataset: DatasetConfig
    paths: PathsConfig
    pairing: PairingConfig
    features: FeaturesConfig
    matching: MatchingConfig
    reconstruction: ReconstructionConfig


def _build_section(section_cls: type, payload: dict[str, Any] | None) -> Any:
    payload = payload or {}
    return section_cls(**payload)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)

    return ExperimentConfig(
        experiment_name=raw.get("experiment_name", "imc2025"),
        runtime=_build_section(RuntimeConfig, raw.get("runtime")),
        dataset=_build_section(DatasetConfig, raw.get("dataset")),
        paths=_build_section(PathsConfig, raw.get("paths")),
        pairing=_build_section(PairingConfig, raw.get("pairing")),
        features=_build_section(FeaturesConfig, raw.get("features")),
        matching=_build_section(MatchingConfig, raw.get("matching")),
        reconstruction=_build_section(ReconstructionConfig, raw.get("reconstruction")),
    )

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from imc2025.config import ExperimentConfig
from imc2025.types import Prediction


def load_samples(config: ExperimentConfig) -> dict[str, list[Prediction]]:
    csv_path = (
        Path(config.paths.train_csv)
        if config.dataset.is_train
        else Path(config.paths.sample_submission_csv)
    )
    df = pd.read_csv(csv_path)

    samples: dict[str, list[Prediction]] = defaultdict(list)
    for _, row in df.iterrows():
        samples[row.dataset].append(
            Prediction(
                image_id=None if config.dataset.is_train else row.image_id,
                dataset=row.dataset,
                filename=row.image,
            )
        )
    return dict(samples)


def get_image_dir(config: ExperimentConfig, dataset_name: str) -> Path:
    split = "train" if config.dataset.is_train else "test"
    return Path(config.paths.data_dir) / split / dataset_name

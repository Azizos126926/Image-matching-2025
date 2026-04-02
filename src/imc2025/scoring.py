from __future__ import annotations

import sys
from pathlib import Path

from imc2025.config import ExperimentConfig


def score_submission(config: ExperimentConfig, submission_path: str | Path):
    metric_module_dir = config.paths.metric_module_dir
    if metric_module_dir not in sys.path:
        sys.path.append(metric_module_dir)

    try:
        import metric
    except Exception as exc:
        raise ImportError(
            "Could not import the competition `metric` module. "
            "Set `paths.metric_module_dir` correctly in your config."
        ) from exc

    return metric.score(
        gt_csv=config.paths.train_csv,
        user_csv=str(submission_path),
        thresholds_csv=config.paths.thresholds_csv,
        mask_csv=None if config.dataset.is_train else str(Path(config.paths.data_dir) / "mask.csv"),
        inl_cf=0,
        strict_cf=-1,
        verbose=True,
    )

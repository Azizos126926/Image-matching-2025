from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path
from time import sleep

import pycolmap

from imc2025.config import ExperimentConfig
from imc2025.types import Prediction
from imc2025.utils.logging import get_logger

LOGGER = get_logger(__name__)


def import_into_colmap(
    image_dir: str | Path,
    feature_dir: str | Path,
    database_path: str | Path,
    metric_module_dir: str,
) -> None:
    """Import feature HDF5 files into a COLMAP database.

    This mirrors the original notebook's dependency on the competition helper
    modules `database.py` and `h5_to_db.py`. The logic is kept behind a clear
    adapter boundary so the rest of the project remains independent from
    Kaggle-specific utility code.
    """
    metric_module_dir = str(metric_module_dir)
    if metric_module_dir not in sys.path:
        sys.path.append(metric_module_dir)

    try:
        from database import COLMAPDatabase
        from h5_to_db import add_keypoints, add_matches
    except Exception as exc:
        raise ImportError(
            "Could not import competition COLMAP helper modules. "
            "Make sure `metric_module_dir` points to the Kaggle IMC utility package."
        ) from exc

    db = COLMAPDatabase.connect(str(database_path))
    db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(
        db,
        str(feature_dir),
        str(image_dir),
        "",
        "simple-pinhole",
        single_camera,
    )
    add_matches(db, str(feature_dir), fname_to_id)
    db.commit()


def run_reconstruction(
    config: ExperimentConfig,
    dataset_name: str,
    image_dir: str | Path,
    feature_dir: str | Path,
    predictions: list[Prediction],
) -> tuple[int, int]:
    image_dir = Path(image_dir)
    feature_dir = Path(feature_dir)
    database_path = feature_dir / "colmap.db"
    output_path = feature_dir / "colmap_rec_aliked"

    if database_path.exists():
        database_path.unlink()

    import_into_colmap(
        image_dir=image_dir,
        feature_dir=feature_dir,
        database_path=database_path,
        metric_module_dir=config.paths.metric_module_dir,
    )

    pycolmap.match_exhaustive(str(database_path))

    mapper_options = pycolmap.IncrementalPipelineOptions()
    mapper_options.min_model_size = config.reconstruction.min_model_size
    mapper_options.max_num_models = config.reconstruction.max_num_models
    output_path.mkdir(parents=True, exist_ok=True)

    maps = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(image_dir),
        output_path=str(output_path),
        options=mapper_options,
    )
    sleep(1)

    filename_to_index = {prediction.filename: idx for idx, prediction in enumerate(predictions)}
    registered = 0

    for map_index, cur_map in maps.items():
        for _, image in cur_map.images.items():
            prediction_idx = filename_to_index[image.name]
            predictions[prediction_idx].cluster_index = map_index
            predictions[prediction_idx].rotation = deepcopy(
                image.cam_from_world.rotation.matrix()
            )
            predictions[prediction_idx].translation = deepcopy(image.cam_from_world.translation)
            registered += 1

    LOGGER.info(
        'Dataset "%s" -> registered %s / %s images with %s clusters',
        dataset_name,
        registered,
        len(predictions),
        len(maps),
    )
    return registered, len(maps)

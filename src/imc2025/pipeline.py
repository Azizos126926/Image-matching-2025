from __future__ import annotations

import gc
from pathlib import Path
from time import time

import torch

from imc2025.config import ExperimentConfig
from imc2025.datasets import get_image_dir, load_samples
from imc2025.features.global_descriptor import DinoV2GlobalDescriptor
from imc2025.features.local import extract_aliked_features, extract_aliked_features_rotated
from imc2025.matching.lightglue import match_with_lightglue
from imc2025.matching.pairs import shortlist_pairs
from imc2025.reconstruction.colmap import run_reconstruction
from imc2025.submission import write_submission
from imc2025.utils.logging import get_logger

LOGGER = get_logger(__name__)


def resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def run_pipeline(config: ExperimentConfig, submission_path: str | Path) -> Path:
    device = resolve_device(config.runtime.device)
    LOGGER.info("Running on device: %s", device)

    samples = load_samples(config)
    descriptor_model = DinoV2GlobalDescriptor(config.paths.dino_model_path, device=device)

    for dataset_name, predictions in samples.items():
        if config.dataset.datasets_to_process and dataset_name not in config.dataset.datasets_to_process:
            LOGGER.info('Skipping dataset "%s"', dataset_name)
            continue

        image_dir = get_image_dir(config, dataset_name)
        images = [str(image_dir / prediction.filename) for prediction in predictions]
        if config.dataset.max_images is not None:
            images = images[: config.dataset.max_images]
            predictions = predictions[: config.dataset.max_images]
            samples[dataset_name] = predictions

        feature_dir = Path(config.paths.work_dir) / "featureout" / dataset_name
        feature_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info('Processing dataset "%s" (%s images)', dataset_name, len(images))
        try:
            t0 = time()
            index_pairs = shortlist_pairs(
                images,
                descriptor_model=descriptor_model,
                similarity_threshold=config.pairing.similarity_threshold,
                min_pairs_per_image=config.pairing.min_pairs_per_image,
                exhaustive_if_less=config.pairing.exhaustive_if_less,
            )
            LOGGER.info("Shortlisting complete: %s pairs in %.2fs", len(index_pairs), time() - t0)

            t0 = time()
            if config.features.use_rotation_augmentation:
                extract_aliked_features_rotated(
                    images,
                    feature_dir=feature_dir,
                    config=config.features,
                    device=device,
                )
            else:
                extract_aliked_features(
                    images,
                    feature_dir=feature_dir,
                    config=config.features,
                    device=device,
                )
            LOGGER.info("Feature extraction complete in %.2fs", time() - t0)

            t0 = time()
            match_with_lightglue(
                images,
                index_pairs=index_pairs,
                feature_dir=feature_dir,
                matching_config=config.matching,
                device=device,
                verbose=config.runtime.verbose,
            )
            LOGGER.info("Matching complete in %.2fs", time() - t0)

            t0 = time()
            run_reconstruction(
                config=config,
                dataset_name=dataset_name,
                image_dir=image_dir,
                feature_dir=feature_dir,
                predictions=predictions,
            )
            LOGGER.info("Reconstruction complete in %.2fs", time() - t0)
            gc.collect()
        except Exception as exc:
            LOGGER.exception('Dataset "%s" failed: %s', dataset_name, exc)

    return write_submission(samples, submission_path=submission_path, is_train=config.dataset.is_train)

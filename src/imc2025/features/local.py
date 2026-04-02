from __future__ import annotations

import os
from pathlib import Path

import cv2
import h5py
import kornia as K
import numpy as np
import torch
from lightglue import ALIKED
from PIL import Image
from tqdm import tqdm

from imc2025.config import FeaturesConfig
from imc2025.utils.image import load_torch_image, pad_to_square


def _build_extractor(
    num_features: int,
    detection_threshold: float,
    resize_to: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ALIKED:
    return ALIKED(
        max_num_keypoints=num_features,
        detection_threshold=detection_threshold,
        resize=resize_to,
    ).eval().to(device, dtype)


def extract_aliked_features(
    image_paths: list[str],
    feature_dir: str | Path,
    config: FeaturesConfig,
    device: torch.device | str,
) -> None:
    device = torch.device(device)
    dtype = torch.float32
    feature_dir = Path(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)

    keypoint_path = feature_dir / "keypoints.h5"
    descriptor_path = feature_dir / "descriptors.h5"

    with h5py.File(keypoint_path, "w") as f_kp, h5py.File(descriptor_path, "w") as f_desc:
        for image_path in tqdm(image_paths, desc="ALIKED features"):
            image_path = str(image_path)
            key = os.path.basename(image_path)

            width, height = Image.open(image_path).size
            resize_this = min(max(width, height), config.resize_to)
            extractor = _build_extractor(
                num_features=config.max_keypoints,
                detection_threshold=config.detection_threshold,
                resize_to=resize_this,
                device=device,
                dtype=dtype,
            )

            with torch.inference_mode():
                image = load_torch_image(image_path, device=device).to(dtype)
                feats = extractor.extract(image)
                keypoints = feats["keypoints"].reshape(-1, 2).detach().cpu().numpy()
                descriptors = feats["descriptors"].reshape(len(keypoints), -1).detach().cpu().numpy()

            f_kp[key] = keypoints
            f_desc[key] = descriptors


def extract_aliked_features_rotated(
    image_paths: list[str],
    feature_dir: str | Path,
    config: FeaturesConfig,
    device: torch.device | str,
) -> None:
    device = torch.device(device)
    dtype = torch.float32
    feature_dir = Path(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)

    rotations = config.rotations or [0, 90]
    with (
        h5py.File(feature_dir / "keypoints_deg.h5", "w") as f_kp_deg,
        h5py.File(feature_dir / "descriptors_deg.h5", "w") as f_desc_deg,
        h5py.File(feature_dir / "offsets.h5", "w") as f_offsets,
        h5py.File(feature_dir / "keypoints.h5", "w") as f_kp_all,
        h5py.File(feature_dir / "descriptors.h5", "w") as f_desc_all,
    ):
        for image_path in tqdm(image_paths, desc="Rotated ALIKED features"):
            image_path = str(image_path)
            key = os.path.basename(image_path)

            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(image_path)
            image = pad_to_square(image)

            f_kp_deg.create_group(key)
            f_desc_deg.create_group(key)
            f_offsets.create_group(key)

            all_keypoints: list[np.ndarray] = []
            all_descriptors: list[np.ndarray] = []
            offset = 0

            working_image = image.copy()
            for angle in rotations:
                if angle == 0:
                    rotated = working_image
                elif angle == 90:
                    rotated = cv2.rotate(working_image, cv2.ROTATE_90_CLOCKWISE)
                else:
                    raise ValueError(
                        f"Only 0 and 90 degree rotations are implemented, got {angle}"
                    )

                image_tensor = K.image_to_tensor(rotated, False).float() / 255.0
                image_tensor = K.color.bgr_to_rgb(image_tensor).to(device).to(dtype)

                extractor = _build_extractor(
                    num_features=config.max_keypoints,
                    detection_threshold=config.detection_threshold,
                    resize_to=config.resize_to,
                    device=device,
                    dtype=dtype,
                )

                with torch.inference_mode():
                    feats = extractor.extract(image_tensor)

                keypoints = feats["keypoints"].squeeze().detach().cpu().numpy()
                descriptors = feats["descriptors"].squeeze().detach().cpu().numpy()

                if descriptors.shape[0] != keypoints.shape[0]:
                    raise ValueError("Mismatch between number of descriptors and keypoints.")

                name = f"{angle}deg"
                f_kp_deg[key][name] = keypoints
                f_desc_deg[key][name] = descriptors
                f_offsets[key][name] = offset

                restored = keypoints.copy()
                h, w = rotated.shape[:2]
                if angle == 90:
                    restored[:, 0] = keypoints[:, 1]
                    restored[:, 1] = w - keypoints[:, 0]

                all_keypoints.append(restored)
                all_descriptors.append(descriptors)
                offset += restored.shape[0]

            f_kp_all[key] = np.concatenate(all_keypoints, axis=0)
            f_desc_all[key] = np.concatenate(all_descriptors, axis=0)

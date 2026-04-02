from __future__ import annotations

import os
from pathlib import Path

import h5py
import kornia.feature as KF
import torch
from tqdm import tqdm

from imc2025.config import MatchingConfig


def match_with_lightglue(
    image_paths: list[str],
    index_pairs: list[tuple[int, int]],
    feature_dir: str | Path,
    matching_config: MatchingConfig,
    device: torch.device | str,
    verbose: bool = False,
) -> None:
    device = torch.device(device)
    feature_dir = Path(feature_dir)

    matcher = KF.LightGlueMatcher(
        "aliked",
        {
            "width_confidence": matching_config.width_confidence,
            "depth_confidence": matching_config.depth_confidence,
            "mp": device.type == "cuda",
        },
    ).eval().to(device)

    with (
        h5py.File(feature_dir / "keypoints.h5", "r") as f_kp,
        h5py.File(feature_dir / "descriptors.h5", "r") as f_desc,
        h5py.File(feature_dir / "matches.h5", "w") as f_match,
    ):
        for idx1, idx2 in tqdm(index_pairs, desc="LightGlue matching"):
            file1 = os.path.basename(image_paths[idx1])
            file2 = os.path.basename(image_paths[idx2])

            keypoints1 = torch.from_numpy(f_kp[file1][...]).to(device)
            keypoints2 = torch.from_numpy(f_kp[file2][...]).to(device)
            desc1 = torch.from_numpy(f_desc[file1][...]).to(device)
            desc2 = torch.from_numpy(f_desc[file2][...]).to(device)

            with torch.inference_mode():
                _, matches = matcher(
                    desc1,
                    desc2,
                    KF.laf_from_center_scale_ori(keypoints1[None]),
                    KF.laf_from_center_scale_ori(keypoints2[None]),
                )

            if len(matches) < matching_config.min_matches:
                continue

            if verbose:
                print(f"{file1}-{file2}: {len(matches)} matches")

            group = f_match.require_group(file1)
            group.create_dataset(file2, data=matches.detach().cpu().numpy().reshape(-1, 2))

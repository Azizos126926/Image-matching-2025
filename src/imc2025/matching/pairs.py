from __future__ import annotations

import numpy as np
import torch

from imc2025.features.global_descriptor import DinoV2GlobalDescriptor


def exhaustive_pairs(image_paths: list[str]) -> list[tuple[int, int]]:
    return [(i, j) for i in range(len(image_paths)) for j in range(i + 1, len(image_paths))]


def shortlist_pairs(
    image_paths: list[str],
    descriptor_model: DinoV2GlobalDescriptor,
    similarity_threshold: float,
    min_pairs_per_image: int,
    exhaustive_if_less: int,
) -> list[tuple[int, int]]:
    if len(image_paths) <= exhaustive_if_less:
        return exhaustive_pairs(image_paths)

    descs = descriptor_model(image_paths)
    dist = torch.cdist(descs, descs, p=2).cpu().numpy()
    mask = dist <= similarity_threshold
    indices = np.arange(len(image_paths))

    pairs: set[tuple[int, int]] = set()
    for src_idx in range(len(image_paths) - 1):
        candidates = indices[mask[src_idx]]
        if len(candidates) < min_pairs_per_image:
            candidates = np.argsort(dist[src_idx])[:min_pairs_per_image]
        for dst_idx in candidates:
            if src_idx == dst_idx:
                continue
            pairs.add(tuple(sorted((int(src_idx), int(dst_idx)))))
    return sorted(pairs)

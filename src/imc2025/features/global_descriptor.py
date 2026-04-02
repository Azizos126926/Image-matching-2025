from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from imc2025.utils.image import load_torch_image


class DinoV2GlobalDescriptor:
    def __init__(self, model_path: str, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        self.processor = AutoImageProcessor.from_pretrained(str(self.model_path))
        self.model = AutoModel.from_pretrained(str(self.model_path)).eval().to(self.device)

    def __call__(self, image_paths: list[str]) -> torch.Tensor:
        descs = []
        for image_path in tqdm(image_paths, desc="DINOv2 descriptors"):
            image = load_torch_image(image_path)
            with torch.inference_mode():
                inputs = self.processor(
                    images=image,
                    return_tensors="pt",
                    do_rescale=False,
                ).to(self.device)
                outputs = self.model(**inputs)
                pooled = outputs.last_hidden_state[:, 1:].max(dim=1)[0]
                desc = F.normalize(pooled, dim=1, p=2)
            descs.append(desc.detach().cpu())
        return torch.cat(descs, dim=0)

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Prediction:
    """Prediction row used for both train and test submissions."""

    image_id: Optional[str]
    dataset: str
    filename: str
    cluster_index: Optional[int] = None
    rotation: Optional[np.ndarray] = None
    translation: Optional[np.ndarray] = None

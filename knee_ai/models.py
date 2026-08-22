from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CaseData:
    """A de-identified image volume and optional anatomical masks.

    Arrays use the project convention ``(ML, SI, AP)``. Spacing is stored in
    millimetres in the same order. Uploaded DICOM data may not have this
    anatomical orientation unless it has first been standardized.
    """

    case_id: str
    image: np.ndarray
    spacing: tuple[float, float, float]
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "uploaded"
    note: str = ""

    def validate(self) -> None:
        if self.image.ndim != 3:
            raise ValueError("Image volume must be three-dimensional.")
        if self.image.size == 0 or self.image.size > 50_000_000:
            raise ValueError("Image volume is empty or exceeds the 50-million-voxel limit.")
        if not np.isfinite(self.image).all():
            raise ValueError("Image volume contains non-finite values.")
        if len(self.spacing) != 3 or any(not np.isfinite(v) or v <= 0 for v in self.spacing):
            raise ValueError("Voxel spacing must contain three positive millimetre values.")
        for name, mask in self.masks.items():
            if mask.shape != self.image.shape:
                raise ValueError(f"Mask '{name}' does not match the image shape.")


@dataclass(frozen=True)
class ThicknessMeasurement:
    region: str
    thickness_mm: float
    slice_index: int
    ap_index: int
    si_start: int
    si_end: int
    samples: int

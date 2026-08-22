from __future__ import annotations

import numpy as np

from .models import CaseData, ThicknessMeasurement


MASK_COLORS = {
    "femur": np.array([38, 198, 218], dtype=np.float32),
    "tibia": np.array([67, 97, 238], dtype=np.float32),
    "meniscus": np.array([246, 103, 92], dtype=np.float32),
}


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    low, high = np.percentile(finite, (1.0, 99.0))
    if high <= low:
        high = low + 1.0
    scaled = np.clip((image - low) / (high - low), 0, 1)
    return (scaled * 255).astype(np.uint8)


def plane_shape(case: CaseData, plane: str) -> tuple[int, int, int]:
    ml, si, ap = case.image.shape
    if plane == "Sagittal":
        return ml, si, ap
    if plane == "Coronal":
        return ap, si, ml
    if plane == "Axial":
        return si, ml, ap
    raise ValueError(f"Unknown plane: {plane}")


def extract_plane(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "Sagittal":
        return volume[index, :, :]
    if plane == "Coronal":
        return np.flipud(volume[:, :, index].T)
    if plane == "Axial":
        return volume[:, index, :]
    raise ValueError(f"Unknown plane: {plane}")


def render_overlay(
    case: CaseData,
    plane: str,
    index: int,
    measurements: list[ThicknessMeasurement] | None = None,
    alpha: float = 0.45,
) -> np.ndarray:
    gray = normalize_to_uint8(extract_plane(case.image, plane, index))
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)

    for name, color in MASK_COLORS.items():
        if name not in case.masks:
            continue
        mask = extract_plane(case.masks[name], plane, index).astype(bool)
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * color

    if plane == "Sagittal" and measurements:
        for measurement in measurements:
            if measurement.slice_index != index:
                continue
            x = measurement.ap_index
            y1, y2 = measurement.si_start, measurement.si_end
            for offset in (-1, 0, 1):
                x_pos = np.clip(x + offset, 0, rgb.shape[1] - 1)
                rgb[y1 : y2 + 1, x_pos] = np.array([255, 221, 64])
            for y in (y1, y2):
                y0, y3 = max(0, y - 2), min(rgb.shape[0], y + 3)
                x0, x3 = max(0, x - 3), min(rgb.shape[1], x + 4)
                rgb[y0:y3, x0:x3] = np.array([255, 221, 64])

    return np.clip(rgb, 0, 255).astype(np.uint8)

from __future__ import annotations

import numpy as np

from .models import CaseData, ThicknessMeasurement


REGIONS = ("Anterior horn", "Body", "Posterior horn")


def _require_mask(case: CaseData, name: str) -> np.ndarray:
    if name not in case.masks or not np.any(case.masks[name]):
        raise ValueError(f"A non-empty '{name}' mask is required for this measurement.")
    return case.masks[name].astype(bool)


def meniscus_thickness(case: CaseData) -> list[ThicknessMeasurement]:
    """Measure representative superior-inferior thickness in three AP thirds.

    This transparent prototype definition is intentionally simple: for every
    medial-lateral slice and AP column, it calculates the occupied SI span, then
    reports the median span in each AP third. A production definition must be
    agreed with radiologists and validated against manual measurements.
    """

    mask = _require_mask(case, "meniscus")
    coordinates = np.argwhere(mask)
    ap_min = int(coordinates[:, 2].min())
    ap_max = int(coordinates[:, 2].max())
    central_slice = int(mask.sum(axis=(1, 2)).argmax())
    edges = np.linspace(ap_min, ap_max + 1, 4).astype(int)
    results: list[ThicknessMeasurement] = []

    for region, start, stop in zip(REGIONS, edges[:-1], edges[1:]):
        samples: list[tuple[float, int, int, int, int]] = []
        for ml_index in range(mask.shape[0]):
            for ap_index in range(start, stop):
                si_indices = np.flatnonzero(mask[ml_index, :, ap_index])
                if si_indices.size:
                    span_pixels = int(si_indices[-1] - si_indices[0] + 1)
                    span_mm = span_pixels * case.spacing[1]
                    samples.append(
                        (span_mm, ml_index, ap_index, int(si_indices[0]), int(si_indices[-1]))
                    )
        if not samples:
            raise ValueError(f"No measurable meniscus columns were found in {region}.")
        target = float(np.median([sample[0] for sample in samples]))
        representative = min(
            samples,
            key=lambda sample: (abs(sample[0] - target), abs(sample[1] - central_slice)),
        )
        results.append(
            ThicknessMeasurement(
                region=region,
                thickness_mm=round(target, 2),
                slice_index=representative[1],
                ap_index=representative[2],
                si_start=representative[3],
                si_end=representative[4],
                samples=len(samples),
            )
        )
    return results


def meniscus_volume_mm3(case: CaseData) -> float:
    mask = _require_mask(case, "meniscus")
    return round(float(mask.sum() * np.prod(case.spacing)), 1)


def meniscus_extrusion_mm(case: CaseData) -> float:
    """Return medial mask extension beyond the aligned tibial mask boundary."""

    meniscus = _require_mask(case, "meniscus")
    tibia = _require_mask(case, "tibia")
    meniscus_min_ml = int(np.argwhere(meniscus)[:, 0].min())
    tibia_min_ml = int(np.argwhere(tibia)[:, 0].min())
    return round(max(0, tibia_min_ml - meniscus_min_ml) * case.spacing[0], 2)


def bone_dimensions(case: CaseData) -> dict[str, float]:
    """Calculate aligned mask extents used by the demonstrator.

    The result is not a surgical resection-plane measurement. Real deployment
    must standardize orientation and use validated anatomical landmarks.
    """

    result: dict[str, float] = {}
    for structure, label in (("femur", "femoral"), ("tibia", "tibial")):
        mask = _require_mask(case, structure)
        coordinates = np.argwhere(mask)
        ml_extent = (coordinates[:, 0].max() - coordinates[:, 0].min() + 1) * case.spacing[0]
        ap_extent = (coordinates[:, 2].max() - coordinates[:, 2].min() + 1) * case.spacing[2]
        result[f"{label}_ml_mm"] = round(float(ml_extent), 2)
        result[f"{label}_ap_mm"] = round(float(ap_extent), 2)
    return result


def measurement_summary(case: CaseData) -> dict[str, float]:
    thickness = meniscus_thickness(case)
    values = {m.region: m.thickness_mm for m in thickness}
    bones = bone_dimensions(case)
    return {
        "anterior_thickness_mm": values["Anterior horn"],
        "body_thickness_mm": values["Body"],
        "posterior_thickness_mm": values["Posterior horn"],
        "meniscus_volume_mm3": meniscus_volume_mm3(case),
        "meniscus_extrusion_mm": meniscus_extrusion_mm(case),
        **bones,
    }

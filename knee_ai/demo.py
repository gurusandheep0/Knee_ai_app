from __future__ import annotations

import numpy as np

from .models import CaseData


DEMO_PROFILES = {
    "Demo 01 · Reference morphology": {
        "case_id": "DEMO-001",
        "seed": 11,
        "age": 34,
        "sex": "Female",
        "oa_status": "Non-OA",
        "joint_shift": 0,
        "meniscus_scale": 1.00,
        "extrusion": 0,
        "irregularity": 0.0,
    },
    "Demo 02 · OA-labelled variation": {
        "case_id": "DEMO-002",
        "seed": 23,
        "age": 59,
        "sex": "Male",
        "oa_status": "OA",
        "joint_shift": 3,
        "meniscus_scale": 0.86,
        "extrusion": 4,
        "irregularity": 1.3,
    },
    "Demo 03 · Advanced OA-labelled variation": {
        "case_id": "DEMO-003",
        "seed": 47,
        "age": 72,
        "sex": "Female",
        "oa_status": "OA",
        "joint_shift": 6,
        "meniscus_scale": 0.72,
        "extrusion": 7,
        "irregularity": 2.4,
    },
}


def demo_case(profile_name: str) -> CaseData:
    """Generate a deterministic, explicitly synthetic 3-D knee-like volume."""

    if profile_name not in DEMO_PROFILES:
        raise KeyError(f"Unknown demo profile: {profile_name}")
    p = DEMO_PROFILES[profile_name]
    shape = (64, 192, 192)  # medial-lateral, superior-inferior, anterior-posterior
    spacing = (1.10, 0.65, 0.65)
    ml, si, ap = np.ogrid[: shape[0], : shape[1], : shape[2]]

    # Distal femur and proximal tibia are represented by smooth synthetic masks.
    femur = (
        ((ml - 32.0) / 29.0) ** 2
        + ((si - (58.0 + p["joint_shift"] * 0.35)) / 34.0) ** 2
        + ((ap - 98.0) / 47.0) ** 2
        <= 1.0
    )
    tibia = (
        ((ml - 35.0) / 29.0) ** 2
        + ((si - (137.0 - p["joint_shift"] * 0.35)) / 31.0) ** 2
        + ((ap - 96.0) / 40.0) ** 2
        <= 1.0
    )

    # A curved wedge approximates medial-meniscus morphology for demonstration.
    ap_norm = (ap - 96.0) / 52.0
    curve = 101.0 + p["joint_shift"] + 4.5 * ap_norm**2
    horn_boost = 4.2 * np.abs(ap_norm) ** 1.7
    central_loss = 2.0 * np.exp(-((ap_norm / 0.30) ** 2))
    waviness = p["irregularity"] * np.sin(ap / 7.0)
    half_thickness = np.maximum(
        2.0,
        (5.2 + horn_boost - central_loss + waviness) * p["meniscus_scale"],
    )
    medial_center = 15.0 - p["extrusion"]
    ml_radius = 8.0
    meniscus = (
        (np.abs(ap_norm) <= 1.0)
        & (np.abs(si - curve) <= half_thickness)
        & (np.abs(ml - medial_center) <= ml_radius * np.sqrt(np.maximum(0, 1 - ap_norm**2)))
    )

    rng = np.random.default_rng(p["seed"])
    image = rng.normal(0.10, 0.025, shape).astype(np.float32)
    # Soft spatial gradients make the viewer look volume-like without pretending
    # that the data are real MR acquisitions.
    image += (0.035 * (si / shape[1]) + 0.018 * np.cos(ap / 18.0)).astype(np.float32)
    image[femur] += 0.36 + rng.normal(0, 0.028, int(femur.sum())).astype(np.float32)
    image[tibia] += 0.32 + rng.normal(0, 0.025, int(tibia.sum())).astype(np.float32)
    image[meniscus] += 0.46 + rng.normal(0, 0.020, int(meniscus.sum())).astype(np.float32)

    # Add narrow bright synthetic cortical edges for visual separation.
    femur_inner = (
        ((ml - 32.0) / 27.0) ** 2
        + ((si - (58.0 + p["joint_shift"] * 0.35)) / 32.0) ** 2
        + ((ap - 98.0) / 45.0) ** 2
        <= 1.0
    )
    tibia_inner = (
        ((ml - 35.0) / 27.0) ** 2
        + ((si - (137.0 - p["joint_shift"] * 0.35)) / 29.0) ** 2
        + ((ap - 96.0) / 38.0) ** 2
        <= 1.0
    )
    image[femur & ~femur_inner] += 0.10
    image[tibia & ~tibia_inner] += 0.09
    image = np.clip(image, 0.0, 1.0)

    case = CaseData(
        case_id=p["case_id"],
        image=image,
        spacing=spacing,
        masks={
            "femur": femur.astype(np.uint8),
            "tibia": tibia.astype(np.uint8),
            "meniscus": meniscus.astype(np.uint8),
        },
        metadata={
            "age": p["age"],
            "sex": p["sex"],
            "oa_status": p["oa_status"],
            "modality": "Synthetic MRI-like volume",
            "laterality": "Right",
        },
        source="bundled-synthetic-demo",
        note=(
            "Synthetic geometry generated inside the application. It contains no patient data "
            "and must not be used to evaluate clinical performance."
        ),
    )
    case.validate()
    return case

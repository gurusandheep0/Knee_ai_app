import json

import numpy as np

from knee_ai.demo import DEMO_PROFILES, demo_case
from knee_ai.implant import load_implants, match_both_components
from knee_ai.measurements import bone_dimensions, measurement_summary, meniscus_thickness
from knee_ai.reporting import build_report, report_json
from knee_ai.visualization import render_overlay


def test_every_demo_case_produces_valid_outputs(tmp_path):
    for profile_name in DEMO_PROFILES:
        case = demo_case(profile_name)
        assert case.image.shape == case.masks["meniscus"].shape
        assert case.image.dtype == np.float32
        assert set(case.masks) == {"femur", "tibia", "meniscus"}

        thickness = meniscus_thickness(case)
        summary = measurement_summary(case)
        assert len(thickness) == 3
        assert all(2.0 <= item.thickness_mm <= 15.0 for item in thickness)
        assert summary["meniscus_volume_mm3"] > 0
        assert summary["femoral_ml_mm"] > 50
        assert summary["tibial_ap_mm"] > 35


def test_overlay_is_rgb_and_contains_annotations():
    case = demo_case(next(iter(DEMO_PROFILES)))
    thickness = meniscus_thickness(case)
    index = int(case.masks["meniscus"].sum(axis=(1, 2)).argmax())
    image = render_overlay(case, "Sagittal", index, thickness)
    assert image.shape == (*case.image.shape[1:], 3)
    assert image.dtype == np.uint8
    assert np.any(np.all(image == [255, 221, 64], axis=2))


def test_implant_matching_and_report(project_root):
    case = demo_case(next(iter(DEMO_PROFILES)))
    dimensions = bone_dimensions(case)
    implants = load_implants(project_root / "data" / "implants.csv")
    matches = match_both_components(implants, dimensions)
    assert set(matches) == {"Femoral", "Tibial"}
    assert all(len(frame) == 3 for frame in matches.values())
    assert all(frame.iloc[0]["rank"] == 1 for frame in matches.values())

    thickness = meniscus_thickness(case)
    report = build_report(case, measurement_summary(case), thickness, matches)
    parsed = json.loads(report_json(report))
    assert parsed["case"]["synthetic"] is True
    assert parsed["prototype_notice"]
    assert len(parsed["implant_matches"]["femoral"]) == 3

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_implants(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"manufacturer", "system", "component", "size", "ml_mm", "ap_mm"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Implant database is missing columns: {sorted(missing)}")
    return frame


def rank_implants(
    implants: pd.DataFrame,
    component: str,
    patient_ml_mm: float,
    patient_ap_mm: float,
    limit: int = 3,
) -> pd.DataFrame:
    candidates = implants.loc[implants["component"].str.lower() == component.lower()].copy()
    if candidates.empty:
        raise ValueError(f"No '{component}' components exist in the implant database.")

    candidates["ml_delta_mm"] = candidates["ml_mm"] - patient_ml_mm
    candidates["ap_delta_mm"] = candidates["ap_mm"] - patient_ap_mm
    candidates["mismatch_pct"] = 100 * (
        0.55 * candidates["ml_delta_mm"].abs() / max(patient_ml_mm, 1.0)
        + 0.45 * candidates["ap_delta_mm"].abs() / max(patient_ap_mm, 1.0)
    )
    candidates["fit_band"] = candidates.apply(
        lambda row: "Closest" if abs(row.ml_delta_mm) <= 3 and abs(row.ap_delta_mm) <= 3 else "Review",
        axis=1,
    )
    candidates = candidates.sort_values(
        ["mismatch_pct", "ml_delta_mm", "ap_delta_mm"], key=lambda values: values.abs()
    ).head(limit)
    candidates.insert(0, "rank", range(1, len(candidates) + 1))
    numeric = ["ml_mm", "ap_mm", "ml_delta_mm", "ap_delta_mm", "mismatch_pct"]
    candidates[numeric] = candidates[numeric].round(2)
    return candidates.reset_index(drop=True)


def match_both_components(
    implants: pd.DataFrame, dimensions: dict[str, float]
) -> dict[str, pd.DataFrame]:
    return {
        "Femoral": rank_implants(
            implants,
            "femoral",
            dimensions["femoral_ml_mm"],
            dimensions["femoral_ap_mm"],
        ),
        "Tibial": rank_implants(
            implants,
            "tibial",
            dimensions["tibial_ml_mm"],
            dimensions["tibial_ap_mm"],
        ),
    }

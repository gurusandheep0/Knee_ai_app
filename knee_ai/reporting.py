from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .models import CaseData, ThicknessMeasurement


def build_report(
    case: CaseData,
    summary: dict[str, float],
    thickness: list[ThicknessMeasurement],
    matches: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prototype_notice": (
            "Research demonstrator only. Not for diagnosis, treatment, or autonomous implant selection."
        ),
        "case": {
            "case_id": case.case_id,
            "source": case.source,
            "synthetic": case.source == "bundled-synthetic-demo",
            "spacing_mm": list(case.spacing),
            "shape": list(case.image.shape),
            "metadata": case.metadata,
            "note": case.note,
        },
        "measurements": summary,
        "meniscus_regions": [
            {
                "region": item.region,
                "thickness_mm": item.thickness_mm,
                "samples": item.samples,
            }
            for item in thickness
        ],
        "implant_matches": {
            component.lower(): frame.to_dict(orient="records")
            for component, frame in matches.items()
        },
    }


def report_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, default=str)


def measurements_csv(case: CaseData, summary: dict[str, float]) -> str:
    row = {"case_id": case.case_id, **case.metadata, **summary}
    return pd.DataFrame([row]).to_csv(index=False)

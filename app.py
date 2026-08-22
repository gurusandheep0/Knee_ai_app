from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from knee_ai import __version__
from knee_ai.demo import DEMO_PROFILES, demo_case
from knee_ai.implant import load_implants, match_both_components
from knee_ai.io import load_uploaded
from knee_ai.measurements import (
    bone_dimensions,
    measurement_summary,
    meniscus_thickness,
)
from knee_ai.models import CaseData
from knee_ai.reporting import build_report, measurements_csv, report_json
from knee_ai.visualization import plane_shape, render_overlay


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

st.set_page_config(
    page_title="KneeAI · Research Demonstrator",
    page_icon="🦵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink: #162331; --muted: #617386; --brand: #176b87; --accent: #ed6a5a; }
    .stApp { background: linear-gradient(180deg, #f7fbfd 0%, #ffffff 32%); }
    [data-testid="stSidebar"] { background: #112633; }
    [data-testid="stSidebar"] * { color: #edf7fa; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stFileUploader label { color: #edf7fa !important; }
    .hero {
      padding: 1.45rem 1.65rem; border-radius: 20px;
      color: white; background: linear-gradient(125deg, #123e52, #197897 62%, #21a0a0);
      box-shadow: 0 14px 35px rgba(17, 62, 82, .18); margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; font-size: 2.2rem; letter-spacing: -.04em; }
    .hero p { margin: .45rem 0 0; opacity: .9; max-width: 850px; }
    .eyebrow { font-size: .74rem; text-transform: uppercase; letter-spacing: .13em; opacity: .8; }
    .info-card {
      min-height: 145px; background: rgba(255,255,255,.94); padding: 1.15rem;
      border-radius: 16px; border: 1px solid #dce9ee; box-shadow: 0 6px 18px rgba(19,55,70,.05);
    }
    .info-card h3 { color: #173f51; margin: 0 0 .45rem; font-size: 1.04rem; }
    .info-card p { color: #617386; margin: 0; line-height: 1.55; }
    .legend { display:flex; gap:1rem; flex-wrap:wrap; font-size:.86rem; color:#526574; }
    .dot { width:.72rem; height:.72rem; display:inline-block; border-radius:50%; margin-right:.32rem; }
    .notice {
      background:#fff8e8; border:1px solid #f0d58d; border-left:5px solid #dfaa24;
      padding:.8rem 1rem; border-radius:10px; color:#664f12; margin:.6rem 0 1rem;
    }
    .small-note { color:#6b7c89; font-size:.84rem; }
    div[data-testid="stMetric"] { background:white; border:1px solid #dce9ee; padding:.75rem; border-radius:14px; }
    .footer { color:#778895; font-size:.78rem; border-top:1px solid #e5edf0; margin-top:2rem; padding-top:1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_demo(name: str) -> CaseData:
    return demo_case(name)


@st.cache_data(show_spinner=False)
def cached_implants() -> pd.DataFrame:
    return load_implants(DATA_DIR / "implants.csv")


@st.cache_data(show_spinner=False)
def cached_cohort() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "cohort.csv")


def hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">AI-assisted quantitative knee imaging</div>
          <h1>KneeAI</h1>
          <p>Explore medial-meniscus morphology, patient-specific bone measurements,
          and transparent implant-size matching in one local research demonstrator.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_case() -> CaseData | None:
    with st.sidebar:
        st.markdown("## KneeAI")
        st.caption(f"Research demonstrator · v{__version__}")
        source = st.radio("Data source", ("Bundled demo", "Upload scan"))
        case: CaseData | None = None

        if source == "Bundled demo":
            profile_name = st.selectbox("Synthetic case", tuple(DEMO_PROFILES))
            case = cached_demo(profile_name)
            st.success("Ready for analysis")
        else:
            uploaded = st.file_uploader(
                "NPZ, NIfTI, DICOM, or DICOM ZIP",
                type=["npz", "nii", "gz", "dcm", "zip"],
                help="For measurements, NPZ must contain image, femur, tibia, meniscus, and spacing arrays.",
            )
            if uploaded is not None:
                try:
                    with st.spinner("Reading de-identified image data…"):
                        case = load_uploaded(uploaded.getvalue(), uploaded.name)
                    st.success("Scan loaded locally")
                except Exception as exc:
                    st.error(f"Could not load scan: {exc}")

        st.divider()
        st.markdown("**Privacy by design**")
        st.caption(
            "The app runs locally. It does not upload images, copy DICOM patient identifiers, "
            "or call external APIs. Use de-identified data only."
        )
        st.markdown("**Mask colors**")
        st.markdown(
            """
            <div class="legend">
              <span><i class="dot" style="background:#26c6da"></i>Femur</span>
              <span><i class="dot" style="background:#4361ee"></i>Tibia</span>
              <span><i class="dot" style="background:#f6675c"></i>Meniscus</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return case


def home_tab(case: CaseData | None) -> None:
    cols = st.columns(3)
    cards = (
        (
            "01 · Segment and verify",
            "Review femur, tibia, and medial-meniscus masks directly over the source volume before using measurements.",
        ),
        (
            "02 · Quantify morphology",
            "Measure anterior horn, body, and posterior horn thickness in physical millimetres, plus volume and extrusion.",
        ),
        (
            "03 · Rank implant fits",
            "Compare aligned femoral and tibial dimensions with a structured, illustrative implant catalogue.",
        ),
    )
    for column, (title, body) in zip(cols, cards):
        with column:
            st.markdown(
                f'<div class="info-card"><h3>{title}</h3><p>{body}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### Active case")
    if case is None:
        st.info("Choose a bundled case or upload a scan from the sidebar.")
        return
    first, second, third, fourth = st.columns(4)
    first.metric("Case", case.case_id)
    second.metric("Source", "Synthetic demo" if case.source == "bundled-synthetic-demo" else "Local upload")
    third.metric("Volume", " × ".join(str(value) for value in case.image.shape))
    fourth.metric("Masks", f"{len(case.masks)} available")
    st.markdown(f'<div class="notice"><strong>Case note:</strong> {case.note}</div>', unsafe_allow_html=True)

    st.markdown("### Intended workflow")
    st.markdown(
        "Select a case → verify overlays → review physical measurements → inspect ranked component "
        "matches → compare the case with the demonstration cohort → export a structured report."
    )


def viewer(case: CaseData, thickness: list | None) -> None:
    st.markdown("### Image and mask verification")
    controls, display = st.columns([1, 3])
    with controls:
        plane = st.radio("View plane", ("Sagittal", "Coronal", "Axial"), horizontal=False)
        slice_count, _, _ = plane_shape(case, plane)
        if plane == "Sagittal" and "meniscus" in case.masks and case.masks["meniscus"].any():
            default_index = int(case.masks["meniscus"].sum(axis=(1, 2)).argmax())
        elif plane == "Coronal" and "meniscus" in case.masks and case.masks["meniscus"].any():
            default_index = int(case.masks["meniscus"].sum(axis=(0, 1)).argmax())
        elif plane == "Axial" and "meniscus" in case.masks and case.masks["meniscus"].any():
            default_index = int(case.masks["meniscus"].sum(axis=(0, 2)).argmax())
        else:
            default_index = slice_count // 2
        index = st.slider(
            "Slice",
            0,
            slice_count - 1,
            min(default_index, slice_count - 1),
            key=f"slice-{case.case_id}-{plane}",
        )
        alpha = st.slider("Overlay opacity", 0.1, 0.8, 0.45, 0.05)
        st.caption("Yellow lines indicate the prototype thickness locations when their sagittal slice is selected.")

    with display:
        rendered = render_overlay(case, plane, index, measurements=thickness, alpha=alpha)
        st.image(
            rendered,
            caption=f"{plane} view · slice {index + 1} of {slice_count}",
            use_container_width=True,
        )
        st.markdown(
            """
            <div class="legend">
              <span><i class="dot" style="background:#26c6da"></i>Femur</span>
              <span><i class="dot" style="background:#4361ee"></i>Tibia</span>
              <span><i class="dot" style="background:#f6675c"></i>Medial meniscus</span>
              <span><i class="dot" style="background:#ffdd40"></i>Measurement</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def analysis_tab(case: CaseData | None) -> tuple[dict | None, list | None, dict | None]:
    if case is None:
        st.info("Select a case from the sidebar to begin.")
        return None, None, None

    required = {"femur", "tibia", "meniscus"}
    missing = sorted(required.difference(case.masks))
    if missing:
        viewer(case, None)
        st.warning(
            "Measurements are disabled because masks are missing: " + ", ".join(missing) + ". "
            "Upload a pre-segmented NPZ file or integrate a validated ONNX model."
        )
        return None, None, None

    try:
        thickness = meniscus_thickness(case)
        summary = measurement_summary(case)
        dimensions = bone_dimensions(case)
    except ValueError as exc:
        st.error(f"Measurement quality check failed: {exc}")
        viewer(case, None)
        return None, None, None

    metric_cols = st.columns(5)
    metric_cols[0].metric("Anterior horn", f"{summary['anterior_thickness_mm']:.2f} mm")
    metric_cols[1].metric("Body", f"{summary['body_thickness_mm']:.2f} mm")
    metric_cols[2].metric("Posterior horn", f"{summary['posterior_thickness_mm']:.2f} mm")
    metric_cols[3].metric("Meniscus volume", f"{summary['meniscus_volume_mm3']:.0f} mm³")
    metric_cols[4].metric("Medial extrusion", f"{summary['meniscus_extrusion_mm']:.2f} mm")

    viewer(case, thickness)

    st.markdown("### Quantitative output")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Medial meniscus")
        thickness_frame = pd.DataFrame(
            [
                {
                    "Region": item.region,
                    "Thickness (mm)": item.thickness_mm,
                    "Valid columns": item.samples,
                }
                for item in thickness
            ]
        )
        st.dataframe(thickness_frame, hide_index=True, use_container_width=True)
        st.caption(
            "Prototype definition: median superior-inferior mask span within each AP third. "
            "Clinical validation is required before interpreting these values."
        )
    with right:
        st.markdown("#### Aligned bone-mask extents")
        bone_frame = pd.DataFrame(
            [
                {"Structure": "Distal femur", "ML (mm)": dimensions["femoral_ml_mm"], "AP (mm)": dimensions["femoral_ap_mm"]},
                {"Structure": "Proximal tibia", "ML (mm)": dimensions["tibial_ml_mm"], "AP (mm)": dimensions["tibial_ap_mm"]},
            ]
        )
        st.dataframe(bone_frame, hide_index=True, use_container_width=True)
        st.caption(
            "These are aligned bounding-mask extents for the demonstrator, not validated surgical resection-plane measurements."
        )

    demographics = case.metadata
    st.markdown(
        f"**Supplied case context:** age `{demographics.get('age', 'Unknown')}` · "
        f"sex `{demographics.get('sex', 'Unknown')}` · "
        f"OA label `{demographics.get('oa_status', 'Unknown')}` · "
        f"modality `{demographics.get('modality', 'Unknown')}`"
    )
    return summary, thickness, dimensions


def implants_tab(case: CaseData | None, dimensions: dict | None) -> dict | None:
    st.markdown(
        '<div class="notice"><strong>Illustrative catalogue:</strong> component records are synthetic. '
        "The ranking is not a prescription and cannot replace surgical planning.</div>",
        unsafe_allow_html=True,
    )
    if case is None or dimensions is None:
        st.info("Complete a case with femur and tibia masks to calculate component matches.")
        return None

    matches = match_both_components(cached_implants(), dimensions)
    femoral, tibial = st.columns(2)
    for column, component in ((femoral, "Femoral"), (tibial, "Tibial")):
        with column:
            target_ml = dimensions[f"{component.lower()}_ml_mm"]
            target_ap = dimensions[f"{component.lower()}_ap_mm"]
            st.markdown(f"### {component} components")
            st.caption(f"Patient mask extent: ML {target_ml:.2f} mm · AP {target_ap:.2f} mm")
            display_frame = matches[component][
                [
                    "rank",
                    "manufacturer",
                    "system",
                    "size",
                    "ml_mm",
                    "ap_mm",
                    "ml_delta_mm",
                    "ap_delta_mm",
                    "mismatch_pct",
                    "fit_band",
                ]
            ].rename(
                columns={
                    "rank": "Rank",
                    "manufacturer": "Maker",
                    "system": "System",
                    "size": "Size",
                    "ml_mm": "ML mm",
                    "ap_mm": "AP mm",
                    "ml_delta_mm": "ML Δ",
                    "ap_delta_mm": "AP Δ",
                    "mismatch_pct": "Mismatch %",
                    "fit_band": "Band",
                }
            )
            st.dataframe(display_frame, hide_index=True, use_container_width=True)
    st.caption(
        "Ranking score = 55% normalized ML error + 45% normalized AP error. Positive deltas indicate "
        "an implant dimension larger than the measured mask extent."
    )
    return matches


def cohort_tab(case: CaseData | None, summary: dict | None) -> None:
    st.markdown("### Demonstration cohort")
    st.caption(
        "This bundled synthetic table exists to demonstrate analysis and chart behavior. "
        "It cannot establish a medical association."
    )
    cohort = cached_cohort().copy()
    filters = st.columns([1, 1, 2])
    sex_values = sorted(cohort["sex"].unique())
    status_values = sorted(cohort["oa_status"].unique())
    sex_filter = filters[0].multiselect("Sex", sex_values, default=sex_values)
    status_filter = filters[1].multiselect(
        "OA label", status_values, default=status_values
    )
    filtered = cohort[cohort["sex"].isin(sex_filter) & cohort["oa_status"].isin(status_filter)]

    if filtered.empty:
        st.warning("No rows match the selected filters.")
        return
    stat_cols = st.columns(4)
    stat_cols[0].metric("Cases", len(filtered))
    stat_cols[1].metric("Mean age", f"{filtered['age'].mean():.1f}")
    stat_cols[2].metric("Mean body thickness", f"{filtered['body_thickness_mm'].mean():.2f} mm")
    stat_cols[3].metric("OA-labelled", f"{(filtered['oa_status'] == 'OA').mean() * 100:.0f}%")

    first, second = st.columns(2)
    with first:
        box = px.box(
            filtered,
            x="oa_status",
            y="body_thickness_mm",
            color="sex",
            points="all",
            labels={"oa_status": "Supplied OA label", "body_thickness_mm": "Body thickness (mm)", "sex": "Sex"},
            color_discrete_sequence=["#176b87", "#ed6a5a"],
        )
        box.update_layout(margin=dict(l=10, r=10, t=25, b=10), legend_orientation="h")
        st.plotly_chart(box, use_container_width=True)
    with second:
        scatter = px.scatter(
            filtered,
            x="age",
            y="body_thickness_mm",
            color="oa_status",
            symbol="sex",
            hover_data=["case_id"],
            labels={"age": "Age", "body_thickness_mm": "Body thickness (mm)", "oa_status": "OA label"},
            color_discrete_map={"Non-OA": "#21a0a0", "OA": "#ed6a5a"},
        )
        if case is not None and summary is not None:
            age = case.metadata.get("age")
            if isinstance(age, (int, float)):
                scatter.add_scatter(
                    x=[age], y=[summary["body_thickness_mm"]], mode="markers+text",
                    marker=dict(size=15, color="#f7c948", line=dict(width=2, color="#463a11")),
                    text=["Active case"], textposition="top center", name="Active case",
                )
        scatter.update_layout(margin=dict(l=10, r=10, t=25, b=10), legend_orientation="h")
        st.plotly_chart(scatter, use_container_width=True)

    with st.expander("View cohort records"):
        st.dataframe(filtered, hide_index=True, use_container_width=True)


def export_tab(
    case: CaseData | None,
    summary: dict | None,
    thickness: list | None,
    matches: dict | None,
) -> None:
    st.markdown("### Structured output")
    if not all((case, summary, thickness, matches)):
        st.info("Complete the image analysis and implant matching to enable exports.")
        return
    report = build_report(case, summary, thickness, matches)
    json_text = report_json(report)
    st.code(json_text, language="json", line_numbers=True)
    first, second = st.columns(2)
    first.download_button(
        "Download full JSON report",
        json_text,
        file_name=f"{case.case_id.lower()}-kneeai-report.json",
        mime="application/json",
        width="stretch",
    )
    second.download_button(
        "Download measurements CSV",
        measurements_csv(case, summary),
        file_name=f"{case.case_id.lower()}-measurements.csv",
        mime="text/csv",
        width="stretch",
    )
    st.caption("Reports intentionally contain the prototype notice, source type, and synthetic-data flag.")


def main() -> None:
    case = sidebar_case()
    hero()
    st.markdown(
        '<div class="notice"><strong>Research prototype:</strong> Not intended for diagnosis, treatment '
        "decisions, or autonomous implant selection. A qualified professional must verify every result.</div>",
        unsafe_allow_html=True,
    )

    home, analysis, implant, cohort, export = st.tabs(
        ("Overview", "Image analysis", "Implant matching", "Cohort insights", "Export")
    )
    with home:
        home_tab(case)
    with analysis:
        summary, thickness, dimensions = analysis_tab(case)
    with implant:
        matches = implants_tab(case, dimensions)
    with cohort:
        cohort_tab(case, summary)
    with export:
        export_tab(case, summary, thickness, matches)

    st.markdown(
        f'<div class="footer">KneeAI v{__version__} · Local-first hackathon demonstrator · '
        "No network services are required after installation.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .models import CaseData


def _normalize(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        raise ValueError("The uploaded image contains no finite values.")
    low, high = np.percentile(finite, (1, 99))
    if high <= low:
        high = low + 1
    return np.clip((volume - low) / (high - low), 0, 1).astype(np.float32)


def load_npz(data: bytes, filename: str = "uploaded.npz") -> CaseData:
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        if "image" not in archive:
            raise ValueError("NPZ input must contain an 'image' array.")
        image = np.asarray(archive["image"])
        if image.ndim == 2:
            image = image[np.newaxis, ...]
        has_spacing = "spacing" in archive
        spacing_array = np.asarray(archive["spacing"] if has_spacing else [1, 1, 1])
        spacing = tuple(float(v) for v in spacing_array.reshape(-1).tolist())
        masks = {}
        for name in ("femur", "tibia", "meniscus"):
            if name in archive:
                mask = np.asarray(archive[name])
                if mask.ndim == 2:
                    mask = mask[np.newaxis, ...]
                masks[name] = (mask > 0).astype(np.uint8)
        if masks and not has_spacing:
            raise ValueError("NPZ inputs with masks must include physical 'spacing' in millimetres.")

    case = CaseData(
        case_id=Path(filename).stem,
        image=_normalize(image),
        spacing=spacing,
        masks=masks,
        metadata={"modality": "NPZ volume", "oa_status": "Unknown"},
        source="uploaded-npz",
        note="Uploaded arrays are processed locally and are not retained by the application.",
    )
    case.validate()
    return case


def load_nifti(data: bytes, filename: str) -> CaseData:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("NIfTI support requires the nibabel package.") from exc

    suffix = ".nii.gz" if filename.lower().endswith(".nii.gz") else ".nii"
    with tempfile.TemporaryDirectory(prefix="knee-ai-") as temp_dir:
        path = Path(temp_dir) / f"scan{suffix}"
        path.write_bytes(data)
        image_object = nib.load(str(path))
        volume_xyz = np.asarray(image_object.get_fdata(dtype=np.float32))
        if volume_xyz.ndim != 3:
            raise ValueError("Only 3-D NIfTI volumes are supported.")
        spacing_xyz = tuple(float(v) for v in image_object.header.get_zooms()[:3])

    # Convert NIfTI (X,Y,Z) to the project volume convention (Z,Y,X).
    case = CaseData(
        case_id=filename.removesuffix(".gz").removesuffix(".nii"),
        image=_normalize(np.transpose(volume_xyz, (2, 1, 0))),
        spacing=(spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]),
        metadata={"modality": "NIfTI volume", "oa_status": "Unknown"},
        source="uploaded-nifti",
        note="No segmentation masks were supplied; viewing is available, but measurements are disabled.",
    )
    case.validate()
    return case


def _dicom_volume(datasets: list[Any], case_id: str) -> CaseData:
    if not datasets:
        raise ValueError("No readable image slices were found in the DICOM input.")

    def slice_coordinate(dataset: Any) -> float:
        position = getattr(dataset, "ImagePositionPatient", None)
        orientation = getattr(dataset, "ImageOrientationPatient", None)
        if position and len(position) >= 3 and orientation and len(orientation) >= 6:
            row = np.asarray(orientation[:3], dtype=float)
            column = np.asarray(orientation[3:6], dtype=float)
            normal = np.cross(row, column)
            return float(np.dot(np.asarray(position[:3], dtype=float), normal))
        if position and len(position) >= 3:
            return float(position[2])
        return float(getattr(dataset, "InstanceNumber", 0))

    datasets.sort(key=slice_coordinate)
    arrays = []
    for dataset in datasets:
        pixels = dataset.pixel_array.astype(np.float32)
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        arrays.append(pixels * slope + intercept)
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError("DICOM slices have inconsistent dimensions.")
    volume = np.stack(arrays)

    first = datasets[0]
    pixel_spacing = getattr(first, "PixelSpacing", [1.0, 1.0])
    row_spacing, column_spacing = float(pixel_spacing[0]), float(pixel_spacing[1])
    slice_spacing = float(getattr(first, "SpacingBetweenSlices", 0) or 0)
    if slice_spacing <= 0 and len(datasets) > 1:
        slice_spacing = abs(slice_coordinate(datasets[1]) - slice_coordinate(datasets[0]))
    if slice_spacing <= 0:
        slice_spacing = float(getattr(first, "SliceThickness", 1.0))

    case = CaseData(
        case_id=case_id,
        image=_normalize(volume),
        spacing=(slice_spacing, row_spacing, column_spacing),
        metadata={
            "modality": str(getattr(first, "Modality", "DICOM")),
            "oa_status": "Unknown",
            "slices": len(datasets),
        },
        source="uploaded-dicom",
        note=(
            "DICOM patient identifiers were not copied. No segmentation model is bundled, so this "
            "scan is view-only until a validated model supplies masks."
        ),
    )
    case.validate()
    return case


def load_dicom(data: bytes, filename: str) -> CaseData:
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError("DICOM support requires the pydicom package.") from exc
    dataset = pydicom.dcmread(io.BytesIO(data))
    return _dicom_volume([dataset], Path(filename).stem)


def load_dicom_zip(data: bytes, filename: str) -> CaseData:
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError("DICOM support requires the pydicom package.") from exc

    datasets: list[Any] = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        if len(members) > 2_000:
            raise ValueError("DICOM archive exceeds the 2,000-file safety limit.")
        if sum(info.file_size for info in members) > 1_000_000_000:
            raise ValueError("DICOM archive exceeds the 1 GB uncompressed safety limit.")
        for info in members:
            try:
                dataset = pydicom.dcmread(io.BytesIO(archive.read(info)), force=False)
                if hasattr(dataset, "PixelData"):
                    datasets.append(dataset)
            except Exception:
                continue
    return _dicom_volume(datasets, Path(filename).stem)


def load_uploaded(data: bytes, filename: str) -> CaseData:
    lower = filename.lower()
    if lower.endswith(".npz"):
        return load_npz(data, filename)
    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        return load_nifti(data, filename)
    if lower.endswith(".zip"):
        return load_dicom_zip(data, filename)
    if lower.endswith(".dcm"):
        return load_dicom(data, filename)
    raise ValueError("Supported uploads are NPZ, NIfTI (.nii/.nii.gz), DICOM, and DICOM ZIP.")

import io

import numpy as np
import pytest

from knee_ai.io import load_npz


def npz_bytes(**arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def test_load_presegmented_npz():
    image = np.arange(2 * 8 * 10, dtype=np.float32).reshape(2, 8, 10)
    femur = image > 120
    tibia = (image > 30) & (image < 70)
    meniscus = (image > 80) & (image < 100)
    case = load_npz(
        npz_bytes(
            image=image,
            spacing=np.array([1.2, 0.7, 0.7]),
            femur=femur,
            tibia=tibia,
            meniscus=meniscus,
        ),
        "sample.npz",
    )
    assert case.image.shape == image.shape
    assert case.spacing == (1.2, 0.7, 0.7)
    assert set(case.masks) == {"femur", "tibia", "meniscus"}


def test_npz_rejects_missing_image():
    with pytest.raises(ValueError, match="image"):
        load_npz(npz_bytes(spacing=np.ones(3)))


def test_npz_rejects_masks_without_physical_spacing():
    with pytest.raises(ValueError, match="spacing"):
        load_npz(npz_bytes(image=np.ones((2, 4, 5)), meniscus=np.ones((2, 4, 5))))

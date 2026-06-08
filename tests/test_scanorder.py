from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"
sys.path.insert(0, str(SOURCE_DIR))

from polyagammadensity import Mixin2D


def test_image_to_scanorder_uses_explicit_row_major_order_for_2_by_3():
    image = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    scan = Mixin2D.image_to_scanorder(image)

    assert np.array_equal(scan, np.array([1, 2, 3, 4, 5, 6]))


def test_scanorder_to_image_uses_explicit_row_major_order_for_3_by_2():
    mixin = Mixin2D()
    scan = np.array([1, 2, 3, 4, 5, 6])

    image = mixin.scanorder_to_image(scan, n=3, m=2)

    assert np.array_equal(
        image,
        np.array(
            [
                [1, 2],
                [3, 4],
                [5, 6],
            ]
        ),
    )


def test_scanorder_roundtrip_preserves_rectangular_image():
    mixin = Mixin2D()
    image = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
        ]
    )

    scan = Mixin2D.image_to_scanorder(image)
    restored = mixin.scanorder_to_image(scan, n=2, m=3)

    assert np.array_equal(restored, image)


def test_image_to_scanorder_rejects_non_2d_input():
    with pytest.raises(AssertionError):
        Mixin2D.image_to_scanorder(np.array([1, 2, 3]))

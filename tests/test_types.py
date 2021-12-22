#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import Any, Dict, Iterable

import numpy as np
import pytest
from pydicom import DataElement

from dicom_utils.types import WINDOW_CENTER, WINDOW_WIDTH, ImageType, SimpleImageType, Window


def get_simple_image_type_test_cases():
    """Seen IMAGE_TYPE Fields:

    2D:
        ['ORIGINAL', 'PRIMARY']
        ['DERIVED', 'PRIMARY']
        ['ORIGINAL', 'PRIMARY', '', '', '', '', '', '', '150000']
        ['DERIVED', 'PRIMARY', 'POST_CONTRAST', 'SUBTRACTION', '', '', '', '', '50000']
        ['ORIGINAL', 'PRIMARY', 'POST_PROCESSED', '', '', '', '', '', '50000']
        ['DERIVED', 'PRIMARY', 'TOMO_PROJ', 'RIGHT', '', '', '', '', '150000'] (may be s-view, but no marker on image)
        ['DERIVED', 'SECONDARY']
        ['DERIVED', 'PRIMARY', 'TOMO_2D', 'LEFT', '', '', '', '', '150000']
        ['DERIVED', 'PRIMARY', 'TOMO_2D', 'RIGHT', '', '', '', '', '150000']

    S-View:
        ['DERIVED', 'PRIMARY', '', '', '', '', '', '', '150000']
        ['DERIVED', 'PRIMARY', 'TOMO', 'GENERATED_2D', '', '', '', '', '150000']
        ['DERIVED', 'PRIMARY', 'TOMOSYNTHESIS', 'GENERATED_2D', '', '', '', '', '150000']

    TOMO:
        ['DERIVED', 'PRIMARY', 'TOMOSYNTHESIS', 'NONE', '', '', '', '', '150000']

    """
    cases = []
    default: Dict[str, Any] = {"pixels": "ORIGINAL", "exam": "PRIMARY"}

    # 2D

    # ['ORIGINAL', 'PRIMARY']
    d = deepcopy(default)
    _ = pytest.param(d, SimpleImageType.NORMAL, id="2d-1")
    cases.append(_)

    # ['DERIVED', 'PRIMARY']
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED"))
    _ = pytest.param(d, SimpleImageType.NORMAL, id="2d-2")
    cases.append(_)

    # ['ORIGINAL', 'PRIMARY', '', '', '', '', '', '', '150000']
    d = deepcopy(default)
    d.update(dict(pixels="ORIGINAL", extras=["", "", "", "", "", "150000"]))
    _ = pytest.param(d, SimpleImageType.NORMAL, id="2d-3")
    cases.append(_)

    # ['DERIVED', 'PRIMARY', 'POST_CONTRAST', 'SUBTRACTION', '', '', '', '', '50000']
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", flavor="POST_CONTRAST", extras=["SUBTRACTION", "", "", "50000"]))
    _ = pytest.param(d, SimpleImageType.NORMAL, id="2d-4")
    cases.append(_)

    # ['ORIGINAL', 'PRIMARY', 'POST_PROCESSED', '', '', '', '', '', '50000']
    d = deepcopy(default)
    d.update(dict(pixels="ORIGINAL", flavor="POST_PROCESSED", extras=["", "", "50000"]))
    _ = pytest.param(d, SimpleImageType.NORMAL, id="2d-5")
    cases.append(_)

    # ['DERIVED', 'PRIMARY', 'TOMO_PROJ', 'RIGHT', '', '', '', '', '150000'] (may be s-view, but no marker on image)
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", flavor="TOMO_PROJ", extras=["RIGHT", "", "50000"]))
    _ = pytest.param(d, SimpleImageType.NORMAL, id="2d-6")
    cases.append(_)

    # ['DERIVED', 'SECONDARY']
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", exam="SECONDARY"))
    _ = pytest.param(d, SimpleImageType.NORMAL, id="2d-7")
    cases.append(_)

    # ['DERIVED', 'PRIMARY', 'TOMO_2D', 'LEFT', '', '', '', '', '150000']
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", flavor="TOMO_2D", extras=["LEFT", "", "", "", "150000"]))
    _ = pytest.param(d, SimpleImageType.NORMAL, id="2d-8")
    cases.append(_)

    # ['DERIVED', 'PRIMARY', 'TOMO_2D', 'RIGHT', '', '', '', '', '150000']
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", flavor="TOMO_2D", extras=["RIGHT", "", "", "", "150000"]))
    _ = pytest.param(d, SimpleImageType.NORMAL, id="2d-9")
    cases.append(_)

    # S-VIEW

    # ['DERIVED', 'PRIMARY', 'TOMO', 'GENERATED_2D', '', '', '', '', '150000']
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", flavor="TOMO", extras=["GENERATED_2D", "", "", "", "150000"]))
    _ = pytest.param(d, SimpleImageType.SVIEW, id="sview-1")
    cases.append(_)

    # ['DERIVED', 'PRIMARY', 'TOMOSYNTHESIS', 'GENERATED_2D', '', '', '', '', '150000']
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", flavor="TOMOSYNTHESIS", extras=["GENERATED_2D", "", "", "", "150000"]))
    _ = pytest.param(d, SimpleImageType.SVIEW, id="sview-2")
    cases.append(_)

    # TOMO

    # ['DERIVED', 'PRIMARY', 'TOMOSYNTHESIS', 'NONE', '', '', '', '', '150000']
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", NumberOfFrames=10, flavor="TOMOSYNTHESIS", extras=["NONE", "", "", "150000"]))
    _ = pytest.param(d, SimpleImageType.TOMO, id="tomo-1")
    cases.append(_)

    return cases


class TestImageType:
    def test_from_dicom(self, dicom_object):
        img_type = ImageType.from_dicom(dicom_object)
        assert img_type.pixels == "ORIGINAL"
        assert img_type.exam == "PRIMARY"
        assert img_type.flavor == "AXIAL"
        assert img_type.NumberOfFrames is None
        assert img_type.model == "RHAPSODE"

    @pytest.mark.parametrize("kwargs,expected", get_simple_image_type_test_cases())
    def test_to_simple_image_type(self, kwargs, expected):
        img_type = ImageType(**kwargs)
        simple_img_type = img_type.to_simple_image_type()
        assert simple_img_type == expected


class TestWindow:
    @pytest.mark.parametrize(
        "center,width",
        [
            pytest.param(None, None),
            pytest.param(DataElement(WINDOW_CENTER, "DS", 512), None),
            pytest.param(None, DataElement(WINDOW_WIDTH, "DS", 512)),
            pytest.param(DataElement(WINDOW_CENTER, "DS", 512), DataElement(WINDOW_WIDTH, "DS", 256)),
            pytest.param(DataElement(WINDOW_CENTER, "DS", 256), DataElement(WINDOW_WIDTH, "DS", 512)),
            pytest.param(
                DataElement(WINDOW_CENTER, "DS", [100, 200, 300]), DataElement(WINDOW_WIDTH, "DS", [200, 300, 400])
            ),
        ],
    )
    def test_from_dicom(self, dicom_object, center, width):
        if center is not None:
            dicom_object[WINDOW_CENTER] = center
        if width is not None:
            dicom_object[WINDOW_WIDTH] = width

        window = Window.from_dicom(dicom_object)

        if center is not None and width is not None:
            assert window.center == center.value if not isinstance(center.value, Iterable) else center[0]
            assert window.width == width.value if not isinstance(width.value, Iterable) else width[0]
        else:
            pixels = dicom_object.pixel_array
            assert window.center == (pixels.max() - pixels.min()) // 2 + pixels.min()
            assert window.width == pixels.max() - pixels.min()

    def test_repr(self):
        window = Window(100, 300)
        print(window)

    @pytest.mark.parametrize(
        "center,width,expected",
        [
            pytest.param(200, 100, 150),
            pytest.param(100, 200, 0),
            pytest.param(100, 300, 0),
        ],
    )
    def test_lower_bound(self, center, width, expected):
        window = Window(center, width)
        assert window.lower_bound == expected

    @pytest.mark.parametrize(
        "center,width,expected",
        [
            pytest.param(200, 100, 250),
            pytest.param(100, 200, 200),
            pytest.param(100, 300, 250),
        ],
    )
    def test_upper_bound(self, center, width, expected):
        window = Window(center, width)
        assert window.upper_bound == expected

    @pytest.mark.parametrize(
        "center,width",
        [
            pytest.param(200, 100),
            pytest.param(100, 200),
            pytest.param(100, 300),
        ],
    )
    def test_apply(self, center, width):
        np.random.seed(42)
        pixels = np.random.random(100).reshape(10, 10)
        pixels = (pixels * 1024).astype(np.uint16)
        window = Window(center, width)

        window_pixels = window.apply(pixels)
        assert (window_pixels >= 0).all()
        assert (window_pixels <= window.width).all()
        assert (window_pixels[pixels <= window.lower_bound] == 0).all()
        assert (window_pixels[pixels >= window.upper_bound] == window.upper_bound - window.lower_bound).all()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import Any, Dict

import pytest

from dicom_utils.types import ImageType, SimpleImageType


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
        assert img_type.NumberOfFrames == 1
        assert img_type.model == "RHAPSODE"

    @pytest.mark.parametrize("kwargs,expected", get_simple_image_type_test_cases())
    def test_to_simple_image_type(self, kwargs, expected):
        img_type = ImageType(**kwargs)
        simple_img_type = img_type.to_simple_image_type()
        assert simple_img_type == expected

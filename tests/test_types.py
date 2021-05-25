#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import Any, Dict

import pytest

from dicom_utils.types import ImageType, get_simple_image_type


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
    _ = pytest.param(d, ImageType.NORMAL, id="2d-1")
    cases.append(_)

    # ['DERIVED', 'PRIMARY']
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED"))
    _ = pytest.param(d, ImageType.NORMAL, id="2d-2")
    cases.append(_)

    # ['ORIGINAL', 'PRIMARY', '', '', '', '', '', '', '150000']
    d = deepcopy(default)
    d.update(dict(pixels="ORIGINAL", extras=["", "", "", "", "", "150000"]))
    _ = pytest.param(d, ImageType.NORMAL, id="2d-3")
    cases.append(_)

    # ['DERIVED', 'PRIMARY', 'POST_CONTRAST', 'SUBTRACTION', '', '', '', '', '50000']
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", flavor="POST_CONTRAST", extras=["SUBTRACTION", "", "", "50000"]))
    _ = pytest.param(d, ImageType.NORMAL, id="2d-4")
    cases.append(_)

    # ['ORIGINAL', 'PRIMARY', 'POST_PROCESSED', '', '', '', '', '', '50000']
    d = deepcopy(default)
    d.update(dict(pixels="ORIGINAL", flavor="POST_PROCESSED", extras=["", "", "50000"]))
    _ = pytest.param(d, ImageType.NORMAL, id="2d-5")
    cases.append(_)

    # ['DERIVED', 'PRIMARY', 'TOMO_PROJ', 'RIGHT', '', '', '', '', '150000'] (may be s-view, but no marker on image)
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", flavor="TOMO_PROJ", extras=["RIGHT", "", "50000"]))
    _ = pytest.param(d, ImageType.NORMAL, id="2d-6")
    cases.append(_)

    # ['DERIVED', 'SECONDARY']
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", exam="SECONDARY"))
    _ = pytest.param(d, ImageType.NORMAL, id="2d-7")
    cases.append(_)

    # ['DERIVED', 'PRIMARY', 'TOMO_2D', 'LEFT', '', '', '', '', '150000']
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", flavor="TOMO_2D", extras=["LEFT", "", "", "", "150000"]))
    _ = pytest.param(d, ImageType.NORMAL, id="2d-8")
    cases.append(_)

    # ['DERIVED', 'PRIMARY', 'TOMO_2D', 'RIGHT', '', '', '', '', '150000']
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", flavor="TOMO_2D", extras=["RIGHT", "", "", "", "150000"]))
    _ = pytest.param(d, ImageType.NORMAL, id="2d-9")
    cases.append(_)

    # S-VIEW

    # ['DERIVED', 'PRIMARY', 'TOMO', 'GENERATED_2D', '', '', '', '', '150000']
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", flavor="TOMO", extras=["GENERATED_2D", "", "", "", "150000"]))
    _ = pytest.param(d, ImageType.SVIEW, id="sview-1")
    cases.append(_)

    # ['DERIVED', 'PRIMARY', 'TOMOSYNTHESIS', 'GENERATED_2D', '', '', '', '', '150000']
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", flavor="TOMOSYNTHESIS", extras=["GENERATED_2D", "", "", "", "150000"]))
    _ = pytest.param(d, ImageType.SVIEW, id="sview-2")
    cases.append(_)

    # TOMO

    # ['DERIVED', 'PRIMARY', 'TOMOSYNTHESIS', 'NONE', '', '', '', '', '150000']
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", NumberOfFrames=10, flavor="TOMOSYNTHESIS", extras=["NONE", "", "", "150000"]))
    _ = pytest.param(d, ImageType.TOMO, id="tomo-1")
    cases.append(_)

    return cases


@pytest.mark.parametrize("image_type,expected", get_simple_image_type_test_cases())
def test_get_simple_image_type_from_dict(image_type, expected):
    actual = get_simple_image_type(image_type)
    assert actual == expected

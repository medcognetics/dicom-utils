#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy

import pytest

from dicom_utils.metadata import ImageType, get_simple_image_type
from dicom_utils.types import ImageType


def get_simple_image_type_test_cases():
    """Seen IMAGE_TYPE Fields:

    2D:
        ['ORIGINAL', 'PRIMARY', '', '', '', '', '', '', '150000']
        ['DERIVED', 'PRIMARY', 'POST_CONTRAST', 'SUBTRACTION', '', '', '', '', '50000']

    S-View:
        ['DERIVED', 'PRIMARY', '', '', '', '', '', '', '150000']

    C-View:
        ['DERIVED', 'PRIMARY']

    TOMO:
        ['DERIVED', 'PRIMARY', 'TOMOSYNTHESIS', 'NONE', '', '', '', '', '150000']

    """
    cases = []
    default = {"pixels": "ORIGINAL", "exam": "PRIMARY"}

    d = deepcopy(default)
    _ = pytest.param(d, ImageType.NORMAL, id="2d-1")
    cases.append(_)

    d = deepcopy(default)
    d.update(dict(pixels="DERIVED"))
    _ = pytest.param(d, ImageType.NORMAL, id="2d-2")
    cases.append(_)

    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", flavor="POST_CONTRAST", extras=["SUBTRACTION", "", "", "50000"]))
    _ = pytest.param(d, ImageType.NORMAL, id="2d-3")
    cases.append(_)

    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", flavor="TOMO", extras=["GENERATED_2D"]))
    _ = pytest.param(d, ImageType.SVIEW, id="sview-1")
    cases.append(_)

    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", flavor="TOMO"))
    _ = pytest.param(d, ImageType.TOMO, id="tomo-1")
    cases.append(_)

    return cases


@pytest.mark.parametrize("image_type,expected", get_simple_image_type_test_cases())
def test_get_simple_image_type_from_dict(image_type, expected):
    actual = get_simple_image_type(image_type)
    assert actual == expected

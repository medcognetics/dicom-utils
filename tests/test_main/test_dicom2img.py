#!/usr/bin/env python
# -*- coding: utf-8 -*-
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from dicom_utils.cli.dicom2img import *


@pytest.fixture
def dicom_file():
    pydicom = pytest.importorskip("pydicom")
    return pydicom.data.get_testdata_file("CT_small.dcm")


@pytest.mark.parametrize("out", [None, "foo.png"])
def test_dicom2img(dicom_file, tmp_path, out):
    if out is not None:
        path = Path(tmp_path, out)
        sys.argv = [sys.argv[0], str(dicom_file), "--noblock", "--output", str(path)]
    else:
        path = None
        sys.argv = [
            sys.argv[0],
            str(dicom_file),
            "--noblock",
        ]

    runpy.run_module("dicom_utils.cli.dicom2img", run_name="__main__", alter_sys=True)

    if out is not None:
        assert path is not None
        assert path.is_file()


@pytest.mark.parametrize(
    "images, expected",
    [
        ([np.array([[[[0]], [[0]], [[0]]]])], np.array([[[[0]], [[0]], [[0]]]])),
        ([np.array([[[[0]], [[1]], [[2]]]])], np.array([[[[0]], [[1]], [[2]]]])),
        (
            [np.array([[[[0]], [[0]], [[0]]]]), np.array([[[[1]], [[1]], [[1]]]])],
            np.array([[[[0, 1]], [[0, 1]], [[0, 1]]]]),
        ),
        (
            [np.array([[[[0]], [[0]], [[0]]]]), np.array([[[[1]], [[1]], [[1]]]]), np.array([[[[2]], [[2]], [[2]]]])],
            np.array([[[[0, 1], [2, 0]], [[0, 1], [2, 0]], [[0, 1], [2, 0]]]]),
        ),
        (
            [
                np.array([[[[7, 0, 0]], [[0, 5, 0]], [[0, 0, 8]]]]),
                np.array([[[[1]], [[1]], [[1]]]]),
                np.array([[[[2]], [[2]], [[2]]]]),
            ],
            np.array(
                [
                    [
                        [[7, 0, 0, 1, 0, 0], [2, 0, 0, 0, 0, 0]],
                        [[0, 5, 0, 1, 0, 0], [2, 0, 0, 0, 0, 0]],
                        [[0, 0, 8, 1, 0, 0], [2, 0, 0, 0, 0, 0]],
                    ]
                ]
            ),
        ),
    ],
)
def test_to_collage(images, expected) -> None:
    collage = to_collage(images)
    assert (expected == collage).all()
    assert expected.shape == collage.shape


def test_to_collage_num_image_assertion() -> None:
    with pytest.raises(AssertionError, match="There must be at least one image."):
        to_collage([])


@pytest.mark.parametrize(
    "ndarrays",
    [
        ([np.array([])]),
        ([np.array([[[0]]]), np.array([])]),
    ],
)
def test_to_collage_image_shape_assertion(ndarrays) -> None:
    with pytest.raises(AssertionError, match="The images must have 4 dimensions."):
        to_collage(ndarrays)


@pytest.mark.parametrize(
    "data, bbox",
    [
        ([1, 1, 2, 2], [0, 0, 2, 2]),
        ([6, 6, 6, 10], [2, 2, 10, 10]),
    ],
)
def test_dicom_circle_to_bbox(data, bbox) -> None:
    predicted_bbox = dicom_circle_to_bbox(data)
    assert list(predicted_bbox) == bbox


@pytest.mark.parametrize(
    "data, bbox",
    [
        ([0, 1, 2, 1, 1, 0, 1, 2], [0, 0, 2, 2]),
    ],
)
def test_dicom_ellipse_to_bbox(data, bbox) -> None:
    predicted_bbox = dicom_ellipse_to_bbox(data)
    assert list(predicted_bbox) == bbox


@pytest.mark.parametrize(
    "form, data",
    [
        ("ELLIPSE", [0] * 8),
        ("CIRCLE", [0] * 4),
    ],
)
def test_dicom_trace_to_bbox(form, data):
    dicom_trace_to_bbox(data, form)


@pytest.mark.parametrize(
    "data, expected_shape",
    [(np.array([0] * 60).reshape((3, 4, 5)), (4, 5, 3))],
)
def test_channel_swaps(data, expected_shape):
    intermediate = chw_to_hwc(data)
    assert intermediate.shape == expected_shape
    assert hwc_to_chw(intermediate).shape == data.shape


@pytest.mark.parametrize(
    "data, expected_shape",
    [(np.array([1] * 20).reshape((1, 4, 5)), (1, 3, 4, 5))],
)
def test_to_rgb(data, expected_shape):
    assert to_rgb(data).shape == expected_shape


@pytest.mark.parametrize(
    "data, expected",
    [
        ([0, 1], [0, 255]),
        ([1, 2], [0, 255]),
        ([0, 1, 2], [0, 127, 255]),
        ([0, 0.5, 1], [0, 127, 255]),
    ],
)
def test_to_8bit(data, expected):
    output = to_8bit(np.array(data))
    assert (output == np.array(expected)).all()

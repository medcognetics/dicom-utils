#!/usr/bin/env python
# -*- coding: utf-8 -*-
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from dicom_utils.cli.dicom2img import to_collage


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
        assert path.is_file()


@pytest.mark.parametrize(
    "images, expected",
    [
        ([np.array([[[0]]])], np.array([[[0]]])),
        ([np.array([[[0]]]), np.array([[[1]]])], np.array([[[0, 1]]])),
        ([np.array([[[0]]]), np.array([[[1]]]), np.array([[[2]]])], np.array([[[0, 1], [2, 0]]])),
        (
            [np.array([[[0, 0, 0]]]), np.array([[[1]]]), np.array([[[2]]])],
            np.array([[[0, 0, 0, 1, 0, 0], [2, 0, 0, 0, 0, 0]]]),
        ),
    ],
)
def test_to_collage(images, expected) -> None:
    assert (expected == to_collage(images)).all()


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
    with pytest.raises(AssertionError, match="The images must have 3 channels."):
        to_collage(ndarrays)

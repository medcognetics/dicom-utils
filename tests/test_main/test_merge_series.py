#!/usr/bin/env python
# -*- coding: utf-8 -*-
import runpy
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import pytest
from pydicom.data import get_testdata_file


@pytest.fixture(
    params=[
        "JPEG2000.dcm",
        "CT_small.dcm",
        "JPEG-LL.dcm",
    ]
)
def dicom_filepath(request: Any) -> str:
    filepath = get_testdata_file(request.param)
    assert isinstance(filepath, str), filepath
    return filepath


def test_merge_series(tmp_path: Path, dicom_filepath: str) -> None:
    num_instances = 2
    for i in range(num_instances):
        shutil.copy(dicom_filepath, tmp_path / f"{i}.dcm")

    sys.argv = [sys.argv[0], str(tmp_path)]
    runpy.run_module("dicom_utils.cli.merge_series", run_name="__main__", alter_sys=True)

    ds = pydicom.dcmread(dicom_filepath)
    ds_merged = pydicom.dcmread(tmp_path / "merged-0.dcm")

    chns, rows, cols = ds_merged.pixel_array.shape
    assert ds_merged.NumberOfFrames == num_instances == chns
    assert ds.pixel_array.shape == (rows, cols)

    expected_pixel_array = np.repeat(ds.pixel_array[np.newaxis, :, :], num_instances, axis=0)
    assert np.allclose(ds_merged.pixel_array, expected_pixel_array)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import runpy
import sys
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.data import get_testdata_file
from pydicom.uid import ExplicitVRLittleEndian, RLELossless

from dicom_utils.dicom_factory import DicomFactory


@pytest.fixture
def get_dicom_file(tmp_path):
    def func(compressed: bool = True):
        get_testdata_file("CT_small.dcm")
        dest = Path(tmp_path, "file.dcm")
        fact = DicomFactory(
            NumberOfFrames=32,
            Rows=128,
            Columns=128,
            TransferSyntaxUID=RLELossless if compressed else ExplicitVRLittleEndian,
        )
        dcm = fact()
        dcm.save_as(dest)
        return dest

    return func


@pytest.mark.parametrize("start_compressed", [True, False])
def test_decompress(get_dicom_file, tmp_path, start_compressed):
    dicom_file = get_dicom_file(start_compressed)
    dest = Path(tmp_path, "output.dcm")
    sys.argv = [
        sys.argv[0],
        str(dicom_file),
        str(dest),
    ]
    runpy.run_module("dicom_utils.cli.project", run_name="__main__", alter_sys=True)

    assert dest.is_file()
    with pydicom.dcmread(dest) as dcm:
        assert not dcm.file_meta.TransferSyntaxUID.is_compressed
        assert isinstance(dcm.pixel_array, np.ndarray)

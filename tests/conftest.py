#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import deepcopy
from pathlib import Path
from typing import Final

import numpy as np
import pydicom
import pytest
from pydicom.data import get_testdata_file
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.uid import (
    UID,
    DeflatedExplicitVRLittleEndian,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    RLELossless,
)


def get_rand_bytes(num: int) -> bytes:
    rand_bytes = np.random.default_rng().bytes(num)
    assert isinstance(rand_bytes, bytes)
    return rand_bytes


@pytest.fixture
def dicom_file():
    return get_testdata_file("CT_small.dcm")


@pytest.fixture
def dicom_object(dicom_file):
    with pydicom.dcmread(dicom_file) as dcm:
        yield dcm


@pytest.fixture
def dicom_file_j2k_uint16() -> str:
    filename = get_testdata_file("RG1_J2KR.dcm")
    assert isinstance(filename, str)
    return filename


@pytest.fixture
def dicom_file_j2k_int16() -> str:
    filename = get_testdata_file("JPEG2000.dcm")
    assert isinstance(filename, str)
    return filename


@pytest.fixture
def dicom_file_j2k_3d_uint16() -> str:
    filename = str(Path(__file__).parent / "jpeg2k_3d.dcm")
    assert isinstance(filename, str)
    return filename


@pytest.fixture
def dicom_file_jpl14() -> str:
    # JPEG lossless non-hierarchical (Process 14)
    filename = get_testdata_file("JPEG-LL.dcm")
    assert isinstance(filename, str)
    return filename


@pytest.fixture(params=["dicom_file_j2k_uint16", "dicom_file_j2k_int16", "dicom_file_j2k_3d_uint16"])
def dicom_file_j2k(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def dicom_file_3d(tmp_path, dicom_object_3d):
    path = Path(tmp_path, "CT_small_3D.dcm")
    dicom_object_3d(num_frames=32).save_as(path)
    return path


@pytest.fixture
def dicom_object_3d(dicom_object):
    series = "1.2.345"
    study = "2.3.456"
    dcm = deepcopy(dicom_object)
    source = Path(pydicom.data.get_testdata_files("*CT*")[0])  # type: ignore
    dcm.StudyInstanceUID = study
    dcm.SeriesInstanceUID = series
    dcm.SOPInstanceUID = series
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID("1.2.345")
    file_meta.MediaStorageSOPInstanceUID = UID("2.3.456")

    def func(num_frames, syntax=ExplicitVRLittleEndian):
        file_meta.TransferSyntaxUID = syntax
        old_data_len = len(dcm.PixelData)
        dcm.NumberOfFrames = num_frames

        if syntax.is_compressed:
            new_data = encapsulate([get_rand_bytes(old_data_len) for _ in range(num_frames)], has_bot=False)
            dcm.PixelData = new_data
            dcm.compress(syntax)
        else:
            dcm.PixelData = get_rand_bytes(old_data_len * num_frames)

        dcm.file_meta = file_meta
        return dcm

    return func


@pytest.fixture(
    params=[
        pytest.param(DeflatedExplicitVRLittleEndian, id="DeflatedExplicitVRLittleEndian"),
        pytest.param(ExplicitVRLittleEndian, id="ExplicitVRLittleEndian"),
        pytest.param(ImplicitVRLittleEndian, id="ImplicitVRLittleEndian"),
        pytest.param(RLELossless, id="RLELossless"),
    ]
)
def transfer_syntax(request):
    return request.param


NUM_DICOM_TEST_FILES: Final[int] = 3


@pytest.fixture(params=pydicom.data.get_testdata_files("*rgb*.dcm")[:NUM_DICOM_TEST_FILES])  # type: ignore
def test_dicom(request) -> Dataset:
    return pydicom.dcmread(request.param)

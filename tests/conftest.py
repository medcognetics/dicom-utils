#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path

import pydicom
import pytest
from pydicom.data import get_testdata_file
from pydicom.dataset import FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.uid import (
    UID,
    DeflatedExplicitVRLittleEndian,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    RLELossless,
)


@pytest.fixture
def dicom_file():
    return get_testdata_file("CT_small.dcm")


@pytest.fixture
def dicom_object(dicom_file):
    with pydicom.dcmread(dicom_file) as dcm:
        yield dcm


@pytest.fixture
def dicom_file_3d(tmp_path, dicom_object_3d):
    path = Path(tmp_path, "CT_small_3D.dcm")
    dicom_file_3d.save_as(path)
    return path


@pytest.fixture
def dicom_object_3d():
    series = "1.2.345"
    study = "2.3.456"
    source = Path(pydicom.data.get_testdata_files("*CT*")[0])  # type: ignore
    with pydicom.dcmread(source) as dcm:
        dcm.StudyInstanceUID = study
        dcm.SeriesInstanceUID = series
        dcm.SOPInstanceUID = series
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = UID("1.2.345")
        file_meta.MediaStorageSOPInstanceUID = UID("2.3.456")

        def func(num_frames, syntax=ExplicitVRLittleEndian):
            file_meta.TransferSyntaxUID = syntax
            old_data = dcm.PixelData
            dcm.NumberOfFrames = num_frames

            if syntax.is_compressed:
                new_data = encapsulate([old_data for _ in range(num_frames)], has_bot=False)
                dcm.PixelData = new_data
                dcm.compress(syntax)

            else:
                new_data = b"".join(old_data for _ in range(num_frames))
                dcm.PixelData = new_data

            dcm.file_meta = file_meta
            return dcm

        yield func


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

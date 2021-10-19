import os
from pathlib import Path
from typing import Final

import pydicom
import pytest
from pydicom import Dataset

from dicom_utils.cli.strip import strip_pixel_data, to_stripped_dicom


num_dicom_test_files: Final[int] = 3


@pytest.fixture(params=pydicom.data.get_testdata_files("*rgb*.dcm")[:num_dicom_test_files])  # type: ignore
def test_dicom_path(request) -> Dataset:
    return request.param


def test_strip_pixel_data(test_dicom_path) -> None:
    test_dicom = pydicom.dcmread(test_dicom_path)
    assert "PixelData" in test_dicom
    strip_pixel_data(test_dicom)
    assert "PixelData" not in test_dicom


def test_to_stripped_dicom(test_dicom_path, tmpdir) -> None:
    new_dicom_path = Path(tmpdir) / "output.dcm"
    to_stripped_dicom(test_dicom_path, new_dicom_path)
    assert "PixelData" in pydicom.dcmread(test_dicom_path)
    assert "PixelData" not in pydicom.dcmread(new_dicom_path)
    assert os.path.getsize(new_dicom_path) < os.path.getsize(test_dicom_path)

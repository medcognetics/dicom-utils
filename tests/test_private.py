from pathlib import Path
from typing import Final

import pydicom
import pytest

from dicom_utils.private import (
    MEDCOG_ADDR,
    MEDCOG_NAME,
    PRIVATE_ELEMENTS_DESCRIPTION,
    get_medcog_elements,
    get_year,
    hash_any,
    store_medcog_elements,
)


num_dicom_test_files: Final[int] = 3


@pytest.fixture(params=pydicom.data.get_testdata_files("*rgb*.dcm")[:num_dicom_test_files])  # type: ignore
def dicom_test_file(request) -> Path:
    return Path(request.param)


@pytest.mark.parametrize(
    "test_data",
    [
        (),
        {},
        [],
        "",
        0,
        None,
        1.0,
    ],
)
def test_hash_any(test_data) -> None:
    hash_any(test_data)


def test_store_value_hashes(dicom_test_file) -> None:
    ds = pydicom.dcmread(dicom_test_file)
    medcog_elements = get_medcog_elements(ds)
    store_medcog_elements(ds, medcog_elements)
    block = ds.private_block(MEDCOG_ADDR, MEDCOG_NAME)
    assert block[0].value == PRIVATE_ELEMENTS_DESCRIPTION
    for i, element in enumerate(medcog_elements):
        assert block[i + 1].VR == element.VR
        assert block[i + 1].value == element.value


@pytest.mark.parametrize(
    "date_string, expected_year",
    [
        ("20200101", "2020"),
        ("19901231", "1990"),
        ("", "????"),
    ],
)
def test_get_year(test_dicom, date_string, expected_year) -> None:
    test_dicom.StudyDate = date_string
    assert expected_year == get_year(test_dicom).value

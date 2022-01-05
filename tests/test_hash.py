from pathlib import Path
from typing import Final

import pydicom
import pytest

from dicom_utils.hash import (
    fields_to_be_hashed,
    get_value_hashes,
    hash_any,
    medcog_addr,
    medcog_name,
    private_fields_description,
    store_value_hashes,
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


def test_fields_to_be_hashed(dicom_test_file) -> None:
    ds = pydicom.dcmread(dicom_test_file)
    for field in fields_to_be_hashed:
        assert ds.get(field) is not None


def test_store_value_hashes(dicom_test_file) -> None:
    ds = pydicom.dcmread(dicom_test_file)
    hash_values = get_value_hashes(ds)
    store_value_hashes(ds, hash_values)
    block = ds.private_block(medcog_addr, medcog_name)
    assert block[0].value == private_fields_description
    for i, value in enumerate(hash_values):
        assert block[i + 1].value == value

import copy
from typing import Final

import pydicom
import pytest

from dicom_utils.anonymize import *
from dicom_utils.hash import medcog_addr, medcog_name, private_fields_description


num_dicom_test_files: Final[int] = 3


@pytest.fixture(params=pydicom.data.get_testdata_files("*rgb*.dcm")[:num_dicom_test_files])  # type: ignore
def test_dicom(request) -> Dataset:
    return pydicom.dcmread(request.param)


@pytest.mark.parametrize(
    "test_data",
    [
        ("1", 1),
        ("078Y", 78),
        ("090Y", 90),
        ("abcdefgh120ijklmnopqrts", 120),
    ],
)
def test_str_to_first_int(test_data) -> None:
    input_string, expected_int = test_data
    assert expected_int == str_to_first_int(input_string)


@pytest.mark.parametrize(
    "test_data",
    [
        ("1", "001Y"),
        ("000078Y", "078Y"),
        ("90Y", "90Y+"),
        ("abcdefgh120ijklmnopqrts", "90Y+"),
    ],
)
def test_age_to_anonymized_age(test_data) -> None:
    input_string, expected_output = test_data
    assert expected_output == age_to_anonymized_age(input_string)


def test_RuleHandler_init() -> None:
    RuleHandler(lambda x: x)


def test_RuleHandler() -> None:
    ds = pydicom.Dataset()
    tag = 0x00000001
    ds[tag] = pydicom.DataElement(value=b"1", tag=tag, VR="CS")
    handler = RuleHandler(lambda _: "x")
    handler(ds, tag)
    assert ds[tag].value == "x"


def test_anonymize(test_dicom) -> None:
    hash_values = get_value_hashes(test_dicom)

    ds = copy.deepcopy(test_dicom)
    anonymize(ds)

    block = get_medcog_block(ds)
    assert block[0].value == private_fields_description
    for i, value in enumerate(hash_values):
        assert block[i + 1].value == value


def test_is_anonymized(test_dicom) -> None:
    not_medcog_name = medcog_name + " "
    test_dicom.private_block(medcog_addr, not_medcog_name, create=True)
    test_dicom.private_block(medcog_addr, not_medcog_name, create=False)  # Check block exists (i.e. no exception)

    # The non-Medcognetics block we just created should not make us think that the case is anonymized
    assert not is_anonymized(test_dicom)
    anonymize(test_dicom)
    assert is_anonymized(test_dicom)

    with pytest.raises(Exception):
        not_medcog_name = medcog_name + "  "
        # This should not return the MedCognetics block but should raise an exception that the block doesn't exist
        test_dicom.private_block(medcog_addr, not_medcog_name, create=False)


def test_double_anonymization(test_dicom) -> None:
    anonymize(test_dicom)
    with pytest.raises(AssertionError, match="DICOM file is already anonymized"):
        anonymize(test_dicom)

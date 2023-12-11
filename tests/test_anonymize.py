import copy

import pydicom
import pytest

from dicom_utils.anonymize import *
from dicom_utils.private import MEDCOG_ADDR, MEDCOG_NAME, PRIVATE_ELEMENTS_DESCRIPTION


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
def test_anonymize_age(test_data) -> None:
    input_string, expected_output = test_data
    assert expected_output == anonymize_age(input_string)


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
    medcog_elements = get_medcog_elements(test_dicom)

    ds = copy.deepcopy(test_dicom)
    anonymize(ds)

    block = get_medcog_block(ds)
    assert block[0].value == PRIVATE_ELEMENTS_DESCRIPTION
    for i, element in enumerate(medcog_elements):
        assert block[i + 1].VR == element.VR
        assert block[i + 1].value == element.value


def test_is_anonymized(test_dicom) -> None:
    not_medcog_name = MEDCOG_NAME + " "
    test_dicom.private_block(MEDCOG_ADDR, not_medcog_name, create=True)
    test_dicom.private_block(MEDCOG_ADDR, not_medcog_name, create=False)  # Check block exists (i.e. no exception)

    # The non-Medcognetics block we just created should not make us think that the case is anonymized
    assert not is_anonymized(test_dicom)
    anonymize(test_dicom)
    assert is_anonymized(test_dicom)

    with pytest.raises(Exception):
        not_medcog_name = MEDCOG_NAME + "  "
        # This should not return the MedCognetics block but should raise an exception that the block doesn't exist
        test_dicom.private_block(MEDCOG_ADDR, not_medcog_name, create=False)


def test_double_anonymization(test_dicom) -> None:
    anonymize(test_dicom)
    with pytest.raises(AssertionError, match="DICOM file is already anonymized"):
        anonymize(test_dicom)

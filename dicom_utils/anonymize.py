import re
import secrets
import string
from typing import Callable, Dict, Final, Optional, TypeVar

import pydicom
from dicomanonymizer import anonymize_dataset
from pydicom import Dataset

from .private import MEDCOG_NAME, get_medcog_block, get_medcog_elements, store_medcog_elements
from .tags import Tag


T = TypeVar("T")

FAKE_UID: Final[bytes] = b"0.0.000.000000.0000.00.0000000000000000.00000000000.00000000"
UID_LEN: Final[int] = len(FAKE_UID)
PATIENT_ID_PREFIX: Final[str] = "MEDCOG"
PATIENT_ID_LOOKUP: Final[Dict[str, str]] = {}


def fix_bad_fields(raw_elem, **kwargs):
    if raw_elem.tag == Tag.NumberOfFrames.tag_tuple and raw_elem.value is None:
        # Value of "None" is non-conformant
        raw_elem = raw_elem._replace(value=1, length=1)
    elif raw_elem.tag == Tag.IrradiationEventUID.tag_tuple and len(raw_elem.value) > UID_LEN:
        # The DICOM anonymizer doesn't handle a list of UIDs properly
        raw_elem = raw_elem._replace(value=FAKE_UID, length=UID_LEN)

    return raw_elem


pydicom.config.data_element_callback = fix_bad_fields  # type: ignore
pydicom.config.convert_wrong_length_to_UN = True  # type: ignore


def preserve_value(x: T) -> T:
    return x


def str_to_first_int(s: str) -> Optional[int]:
    x = re.findall(r"\d+", s)
    if len(x) > 0:
        return int(x[0])


def anonymize_age(age_str: str) -> str:
    """So few people live into their 90s that an age greater than 89 is considered to be identifying information."""
    age: Optional[int] = str_to_first_int(age_str)
    if age is None:
        return "----"
    elif age > 89:
        return "90Y+"
    else:
        return f"{age:03}Y"


def generate_rand_str(num_chars: int = 32) -> str:
    chars62 = string.ascii_letters + string.digits  # Almost 6-bits
    return "".join(secrets.choice(chars62) for _ in range(num_chars))


def anonymize_patient_id(patient_id: str) -> str:
    if patient_id not in PATIENT_ID_LOOKUP:
        PATIENT_ID_LOOKUP[patient_id] = f"{PATIENT_ID_PREFIX}{generate_rand_str()}"
    return PATIENT_ID_LOOKUP[patient_id]


RuleMap = Dict[Tag, Callable[[T], T]]

RULES: Final[RuleMap] = {
    Tag.PatientID: anonymize_patient_id,
    Tag.PatientAge: anonymize_age,
    Tag.PatientSex: preserve_value,
    Tag.CountryOfResidence: preserve_value,
    Tag.EthnicGroup: preserve_value,
    # Institution names should be OK to keep per the following explanation:
    # "Only names of the individuals associated with the corresponding health
    # information (i.e., the subjects of the records) and of their relatives,
    # employers, and household members must be suppressed. There is no explicit
    # requirement to remove the names of providers or workforce members of the
    # covered entity or business associate."
    # https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html#supress
    Tag.InstitutionName: preserve_value,
    # `DerivationDescription` describes operations with which a DICOM image has been modified,
    # and it can be useful for checking settings that may affect image quality.
    Tag.DerivationDescription: preserve_value,
}


def is_anonymized(ds: Dataset) -> bool:
    try:
        get_medcog_block(ds)
        return True
    except KeyError as e:
        assert str(e) == f"\"Private creator '{MEDCOG_NAME}' not found\""
        return False


def anonymize(ds: Dataset) -> None:
    # anonymize_dataset() deletes private elements
    # so we need to store value hashes in the MedCognetics private elements after anonymization
    assert not is_anonymized(ds), "DICOM file is already anonymized"

    overrides = {tag: rule(getattr(ds, tag.name, "")) for tag, rule in RULES.items()}
    elements = get_medcog_elements(ds)

    anonymize_dataset(ds, delete_private_tags=True)

    store_medcog_elements(ds, elements)
    for tag, value in overrides.items():
        setattr(ds, tag.name, value)

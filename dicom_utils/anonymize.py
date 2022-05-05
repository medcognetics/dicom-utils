import re
from typing import Any, Callable, Dict, Final, Optional, TypeVar

from dicomanonymizer import anonymize_dataset
from pydicom import Dataset

from .private import MEDCOG_NAME, get_medcog_block, get_medcog_elements, store_medcog_elements
from .tags import Tag


T = TypeVar("T")


class RuleHandler:
    def __init__(self, handler: Callable[[Any], Any]) -> None:
        self.handler = handler

    def __call__(self, dataset: Dataset, tag: int) -> Any:
        element = dataset.get(tag)
        if element is not None:
            element.value = self.handler(element.value)


def return_input(x: T) -> T:
    return x


preserve_value: Final[RuleHandler] = RuleHandler(return_input)


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


RuleMap = Dict[Tag, RuleHandler]

rules: Final[RuleMap] = {
    Tag.PatientAge: RuleHandler(anonymize_age),
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
    elements = get_medcog_elements(ds)
    anonymize_dataset(ds, rules)
    store_medcog_elements(ds, elements)

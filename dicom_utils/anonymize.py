import re
from typing import Any, Callable, Dict, Final, Optional, Tuple, TypeVar

from dicomanonymizer import anonymize_dataset
from pydicom import Dataset

from .hash import get_medcog_block, get_value_hashes, medcog_name, store_value_hashes
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


def age_to_anonymized_age(age_str: str) -> str:
    """So few people live into their 90s that an age greater than 89 is considered to be identifying information."""
    age: Optional[int] = str_to_first_int(age_str)
    if age is None:
        return "----"
    elif age > 89:
        return "90Y+"
    else:
        return f"{age:03}Y"


TagTuple = Tuple[int, int]
RuleMap = Dict[Tag, RuleHandler]

rules: Final[RuleMap] = {
    Tag.PatientAge: RuleHandler(age_to_anonymized_age),
}

# TODO these tags are anonymized, but we probably want to keep them
# (0x0010, 0x2150), # Country of Residence
# (0x0010, 0x2160), # Ethnic Group
# (0x0032, 0x4000), # Study Comments (potential case notes)
# (0x0008, 0x103E), # Series Description
# (0x0008, 0x0080), # Institution Name
# (0x0018, 0x1000), # Device Serial Number
# (0x0010, 0x0040), # Patient's Sex


def is_anonymized(ds: Dataset) -> bool:
    try:
        get_medcog_block(ds)
        return True
    except KeyError as e:
        assert str(e) == f"\"Private creator '{medcog_name}' not found\""
        return False


def anonymize(ds: Dataset) -> None:
    # anonymize_dataset() deletes private elements
    # so we need to store value hashes in the MedCognetics private elements after anonymization
    assert not is_anonymized(ds), "DICOM file is already anonymized"
    value_hashes = get_value_hashes(ds)
    anonymize_dataset(ds, rules)
    store_value_hashes(ds, value_hashes)

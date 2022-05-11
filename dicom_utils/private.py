import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Final, List, Optional

from pydicom import Dataset
from pydicom.dataset import PrivateBlock

from .tags import Tag


@dataclass
class MedCogElement:
    value: Any
    VR: str


@dataclass
class MedCogElementCreator:
    name: str
    create_element: Callable[[Dataset], MedCogElement]


class VR(Enum):
    other_byte_string = "OB"
    long_text = "LT"


def get_study_date(ds: Dataset) -> Optional[datetime]:
    if (study_date := ds.get(Tag.StudyDate, None)) and study_date.value:
        return datetime.strptime(study_date.value, DICOM_DATE_FORMAT)


def get_year(ds: Dataset) -> MedCogElement:
    # Elements of dates that are not permitted for disclosure include the day, month, and any other information that
    # is more specific than the year of an event. For instance, the date “January 1, 2009” could not be reported at
    # this level of detail. However, it could be reported in a de-identified data set as “2009”.
    # https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html
    study_date = get_study_date(ds)
    year_str = str(study_date.year) if study_date else "????"
    return MedCogElement(year_str, VR.long_text.value)


def hash_pixel_data(ds: Dataset) -> MedCogElement:
    if pixel_data := ds.get(Tag.PixelData, None):
        value = hash_any(pixel_data)
    else:
        value = b""
    return MedCogElement(value, VR.other_byte_string.value)


MEDCOG_ELEMENT_CREATORS: Final[List[MedCogElementCreator]] = [
    MedCogElementCreator("PixelDataHash", hash_pixel_data),
    MedCogElementCreator("StudyYear", get_year),
]

DICOM_DATE_FORMAT: Final[str] = r"%Y%m%d"
MEDCOG_ADDR: Final[int] = int("0x" + "".join(f"{ord(v):2x}" for v in "MC"), 16)
MEDCOG_NAME: Final[str] = "MedCognetics"
PRIVATE_ELEMENTS_DESCRIPTION: Final[str] = ",".join(f.name for f in MEDCOG_ELEMENT_CREATORS)
assert len(PRIVATE_ELEMENTS_DESCRIPTION) <= 10240, f"Description exceeds maximum length for VR '{VR.long_text.value}'"


def hash_bytes(x: bytes) -> bytes:
    sha256 = hashlib.sha256()
    sha256.update(x)
    return sha256.digest()


def hash_any(value: Any, encoding: str = "utf-8") -> bytes:
    # NOTE: Hashing patient names and other identifying fields may be unsafe
    # due to bute-force or rainbow table attacks.
    value = str(value).encode(encoding)
    return hash_bytes(value)


def get_medcog_elements(ds: Dataset) -> List[MedCogElement]:
    return [field.create_element(ds) for field in MEDCOG_ELEMENT_CREATORS]


def get_medcog_block(ds: Dataset, create: bool = False) -> PrivateBlock:
    return ds.private_block(MEDCOG_ADDR, MEDCOG_NAME, create=create)


def store_medcog_elements(ds: Dataset, field_values: List[MedCogElement]) -> None:
    block = get_medcog_block(ds, create=True)
    block.add_new(0, VR.long_text.value, PRIVATE_ELEMENTS_DESCRIPTION)
    for i, field in enumerate(field_values):
        block.add_new(i + 1, field.VR, field.value)

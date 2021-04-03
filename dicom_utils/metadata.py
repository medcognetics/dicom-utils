#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import datetime as dt
import json
import re
from typing import Any, Dict, Final, Optional

from pydicom import DataElement

from .types import Dicom, ImageType


LINE_PATTERN = re.compile(r"^\((\d{4}), (\d{4})\)\s*(.*\S).*([A-Z]{2}):\s*(.*)$")

GROUPS: Dict[str, str] = {"image": "0028", "media": "0008"}

# fields with values above this limit will be dropped
MAX_FIELD_LENGTH: Final[int] = 100

AGE_TAG = 0x00101010
DOB = 0x00100030
STUDY_DATE = 0x00080020
CONTENT_DATE = 0x00080023
ACQUISITION_DATE = 0x00080022
DENSITY = 0x40101018
IMAGE_TYPE = 0x00080008


def is_inverted(photo_interp: str) -> bool:
    """Checks if pixel value 0 corresponds to white. See DICOM specification for more details."""
    if photo_interp == "MONOCHROME1":
        return True
    elif photo_interp != "MONOCHROME2":
        # I don't think we need to handle any interpretations besides MONOCHROME1
        # and MONOCHROME2 in the case of mammograms.
        raise Exception(f"Unexpected photometric interpretation '{photo_interp}'")
    return False


def drop_empty_tags(dcm: Dicom) -> Dicom:
    r"""Drops empty tags from a DICOM object. This applies only to the in-memory
    DICOM object - no modifications are made to the file on disk.

    Args:
        dcm (:class:`pydicom.FileDataset`):
            The DICOM object to drop empty tags from

    Returns:
        ``dcm`` with all empty tags removed.
    """
    drop_keys = []
    for key in dcm.keys():
        try:
            tag = dcm.get(key)
            if tag.is_empty:
                drop_keys.append(key)
        except AttributeError:
            drop_keys.append(key)
    for k in drop_keys:
        del dcm[k]
    return dcm


def drop_fields_by_length(dcm: Dicom, max_field_length: int = MAX_FIELD_LENGTH, inplace: bool = False) -> Dicom:
    r"""Drops DICOM fields with values with lengths above a threshold, or fields with
    undefined lengths.

    Args:
        dcm (:class:`pydicom.FileDataset`):
            The DICOM object to process

        max_field_length (int):
            Fields above this length will be dropped

        inplace (bool):
            By default, a deep copy of the DICOM object is created. When ``inplace=True``,
            remove fields from the source DICOM object without creating a copy.
    """
    if not inplace:
        dcm = copy.deepcopy(dcm)

    keys = set(dcm.keys())
    for k in keys:
        v = dcm[k]
        if hasattr(v.value, "__len__") and len(v.value) > max_field_length:
            del dcm[k]
        elif v.is_undefined_length:
            del dcm[k]
    return dcm


def add_patient_age(dcm: Dicom, age: Optional[int] = None) -> Optional[int]:
    r"""Attempts to add the specified patient age to a DICOM object.

    An age can be determined using the following order of precedence:
        1. Reading an existing age tag in the DICOM metadata
        2. Computing an age using the date of birth and capture date metadata
        3. Adding the age passed as an ``age`` parameter.

    If ``age`` is given and ``age`` doesn't match the age determined via DICOM
    metadata, the age will not be updated.

    Args:
        dcm (:class:`pydicom.FileDataset`):
            The DICOM object to add an age to

        age (int, optional):
            The age to attempt to add to ``dcm``.

    Returns:
        The determined patient age in years, or ``None`` if no age could be
        determined.
    """
    capture_date = get_date(dcm)

    if AGE_TAG in dcm:
        dcm_age = int(re.sub("[^0-9]", "", dcm[AGE_TAG].value))
    elif DOB in dcm and capture_date is not None:
        dob = dcm[DOB].value
        try:
            dob = dt.datetime.strptime(dob, "%m%d%Y")
            delta = (capture_date - dob).days / 365.25
            dcm_age = round(delta)
        except ValueError:
            dcm_age = age
    else:
        dcm_age = age

    if dcm_age is not None:
        if dcm_age != age and age is not None:
            print(f"Computed age {dcm_age} doesn't match given age {age}. " "Not updating patient age.")
        else:
            dcm[AGE_TAG] = DataElement(AGE_TAG, "AS", f"{dcm_age:03d}Y")
    elif age is not None:
        dcm[AGE_TAG] = DataElement(AGE_TAG, "AS", f"{age:03d}Y")
    return dcm_age


def get_date(dcm: Dicom, fmt: str = "%Y%m%d") -> Optional[dt.datetime]:
    r"""Attempts to parse a collection date from a DICOM object.

    Dates are parsed using the following order of precedence:
        1. CONTENT_DATE tag
        2. ACQUISITION_DATE tag
        3. STUDY_DATE tag

    Returns:
        Parsed date, or ``None`` if no date could be parsed.
    """
    targets = (CONTENT_DATE, ACQUISITION_DATE, STUDY_DATE)
    for target in targets:
        if target in dcm.keys():
            date = dcm[target].value
            date = dt.datetime.strptime(date, fmt)
            return date
    return None


def add_density(dcm: Dicom, density: float) -> str:
    if DENSITY not in dcm.keys():
        dcm[DENSITY] = DataElement(DENSITY, "FL", density)
        return str(density)
    else:
        return str(dcm[DENSITY].value)


def dicom_to_json(dcm: Dicom) -> Dict[str, Any]:
    r"""Converts DICOM metadata into a JSON dictionary."""
    return json.loads(dcm.to_json())


def get_simple_image_type(image_type: Dict[str, Any]) -> ImageType:
    r"""Produces an ImageType value from a dictionary of processed image type
    fields (from :func:`process_image_type`).

    ImageType is determined as follows:
        1. ImageType.TOMO if ``NumberOfFrames > 1``
        2. ImageType.SVIEW if ``pixels == "derived`` and (``"generated" in extras or flavor is not None``)
        3. ImageType.2D otherwise
    """
    # if fields 1 and 2 were missing, we know nothing
    if not image_type:
        return ImageType.UNKNOWN

    pixels = image_type.get("pixels", "").lower()
    flavor = image_type.get("flavor", None)
    extras = image_type.get("extras", None)
    num_frames = int(image_type.get("NumberOfFrames", 1))

    if num_frames > 1:
        return ImageType.TOMO

    if "derived" in pixels:
        if "tomo" in flavor.lower():
            if extras is not None and any("generated" in x.lower() for x in extras):
                return ImageType.SVIEW
            elif "2d" in flavor.lower():
                return ImageType.NORMAL
            return ImageType.TOMO
        elif flavor is not None:
            return ImageType.SVIEW

    return ImageType.NORMAL


def process_image_type(dcm: Dicom) -> Dict[str, Any]:
    r"""Extracts and processes fields that identify the DICOM image type.
    The returned dectionary will be populated with the following items (as available):
        * ``"pixels"`` - First element of the IMAGE_TYPE field.
        * ``"exam"`` - Second element of the IMAGE_TYPE field.
        * ``"flavor"`` - Third element of the IMAGE_TYPE field.
        * ``"extras"`` - Additional IMAGE_TYPE fields if available
        * ``"NumberOfFrames"`` - Frame count (for TOMO images only)
    """
    # http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.24.3.2.html

    result = {}

    if (num_frames := getattr(dcm, "NumberOfFrames", None)) is not None:
        result["NumberOfFrames"] = num_frames

    if IMAGE_TYPE not in dcm.keys():
        return result

    try:
        # fields 1 and 2 should always be present
        image_type = dcm[IMAGE_TYPE].value
        pixels, exam = image_type[:2]
        result["pixels"] = pixels
        result["exam"] = exam

        # there might be a field 3
        if len(image_type) >= 3:
            flavor = image_type[2]
            result["flavor"] = flavor

        if len(image_type) >= 4:
            result["extras"] = image_type[3:]

    except RuntimeError:
        pass

    return result

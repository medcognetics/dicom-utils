#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum
from typing import Any, Dict

from pydicom.dataset import FileDataset


IMAGE_TYPE = 0x00080008

Dicom = FileDataset


def process_image_type(dcm: Dicom) -> Dict[str, Any]:
    r"""Extracts and processes fields that identify the DICOM image type.
    The returned dectionary will be populated with the following items (as available):
        * ``"pixels"`` - First element of the IMAGE_TYPE field.
        * ``"exam"`` - Second element of the IMAGE_TYPE field.
        * ``"flavor"`` - Third element of the IMAGE_TYPE field.
        * ``"extras"`` - Additional IMAGE_TYPE fields if available
        * ``"NumberOfFrames"`` - Frame count (for TOMO images only)
        * ``"model"`` - Manufacturer's model name
    """
    # http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.24.3.2.html

    result: Dict[str, Any] = {}
    result["NumberOfFrames"] = dcm.get("NumberOfFrames", 1)
    if (model := dcm.get(0x00081090, None)) is not None:
        result["model"] = model.value
    else:
        result["model"] = ""

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


class ImageType(Enum):
    UNKNOWN = 0
    NORMAL = 1
    SVIEW = 2
    TOMO = 3

    @staticmethod
    def from_dicom(dcm: Dicom) -> "ImageType":
        keys = process_image_type(dcm)
        return get_simple_image_type(keys)

    def __str__(self) -> str:
        if self is ImageType.UNKNOWN:
            return "unknown"
        elif self is ImageType.NORMAL:
            return "2d"
        elif self is ImageType.SVIEW:
            return "s-view"
        elif self is ImageType.TOMO:
            return "tomo"
        else:
            raise RuntimeError("unknown ImageType value")


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

    pixels = image_type.get("pixels", "").lower() or ""
    flavor = image_type.get("flavor", "").lower() or ""
    exam = image_type.get("exam", "").lower() or ""
    extras = image_type.get("extras", []) or []
    num_frames = int(image_type.get("NumberOfFrames", 1) or 1)
    machine = image_type.get("model", "").lower() or ""

    # very solid rules
    if num_frames > 1:
        return ImageType.TOMO
    if "original" in pixels:
        return ImageType.NORMAL

    # ok rules
    if extras is not None and any("generated_2d" in x.lower() for x in extras):
        return ImageType.SVIEW

    # not good rules
    if pixels == "derived" and exam == "primary" and machine == "fdr-3000aws" and flavor != "post_contrast":
        return ImageType.SVIEW

    return ImageType.NORMAL


__all__ = ["Dicom", "ImageType"]

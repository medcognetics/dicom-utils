#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import sys
from pathlib import Path
from typing import Dict, Final, Optional, Tuple, Union

import numpy as np

from .types import Dicom


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


class NoImageError(Exception):
    pass


def is_native_byteorder(arr: np.ndarray) -> bool:
    r"""Checks if a numpy array has native byte order (Endianness)"""
    array_order = arr.dtype.byteorder
    if array_order in ["=", "|"]:
        return True
    return sys.byteorder == "little" and array_order == "<" or array_order == ">"


def is_inverted(photo_interp: str) -> bool:
    """Checks if pixel value 0 corresponds to white. See DICOM specification for more details."""
    if photo_interp == "MONOCHROME1":
        return True
    elif photo_interp != "MONOCHROME2":
        # I don't think we need to handle any interpretations besides MONOCHROME1
        # and MONOCHROME2 in the case of mammograms.
        raise Exception(f"Unexpected photometric interpretation '{photo_interp}'")
    return False


def invert_color(img: np.ndarray) -> np.ndarray:
    """The maximum value will become the minimum and vice versa"""
    return np.max(img) - img


def has_dicm_prefix(filename: Union[str, Path]) -> bool:
    """DICOM files have a 128 byte preamble followed by bytes 'DICM'."""
    with open(filename, "rb") as f:
        f.seek(128)
        return f.read(4) == b"DICM"


def read_image(dcm: Dicom, stop_before_pixels: bool = False, shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    r"""
    Reads image data from an open DICOM file into a Tensor.

    Args:
        dcm (:class:`pydicom.FileDataset`): DICOM object to load images from
        stop_before_pixels (bool): If true, return randomly generated data
        shape (tuple of ints): Manual shape override when ``stop_before_pixels`` is true

    Shape:
        - Output: :math:`(1, H, W)` or :math:`(1, D, H, W)`
    """
    # some dicoms dont have any image data - raise NoImageError
    for necessary_field in ["Rows", "PhotometricInterpretation"]:
        if shape is None and not hasattr(dcm, necessary_field):
            raise NoImageError()

    if shape is None:
        D, H, W = dcm.get("NumberOfFrames", None), int(dcm.Rows), int(dcm.Columns)
        if D is not None:
            D = int(D)
            dims = (D, H, W)  # type: ignore
        else:
            dims = (H, W)  # type: ignore
    else:
        dims = shape

    # return random pixel data in correct shape when stop_before_pixels=True
    if stop_before_pixels:
        return np.random.randint(0, 2 ** 10, dims)

    data = np.ndarray(dims, dcm.pixel_array.dtype, dcm.pixel_array)

    # in some dicoms, pixel value of 0 indicates white
    if is_inverted(dcm.PhotometricInterpretation):  # type: ignore
        data = invert_color(data)

    # some dicoms have different endianness - convert to native byte order
    if not is_native_byteorder(data):
        data = data.byteswap().newbyteorder()
    assert is_native_byteorder(data)

    # numpy byte order needs to explicitly be native "=" for torch conversion
    if data.dtype.byteorder != "=":
        data = data.newbyteorder("=")

    # torch has no uint16 dtype - cast to int32
    if data.dtype == np.uint16:
        data = data.astype(np.int32)

    return data

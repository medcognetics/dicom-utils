#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from numpy import ndarray

from .types import Dicom


class NoImageError(Exception):
    pass


def is_native_byteorder(arr: ndarray) -> bool:
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


def invert_color(img: ndarray) -> ndarray:
    """The maximum value will become the minimum and vice versa"""
    return np.max(img) - img


def has_dicm_prefix(filename: Union[str, Path]) -> bool:
    """DICOM files have a 128 byte preamble followed by bytes 'DICM'."""
    with open(filename, "rb") as f:
        f.seek(128)
        return f.read(4) == b"DICM"


def uncompressed_dcm_to_pixels(dcm: Dicom, dims: Tuple[int, ...]) -> ndarray:
    """
    Ignore field (0002, 0010) Transfer Syntax UID which should describe pixel encoding/compression
    and instead interpret the pixel data as uncompressed.

    Some mammograms contain an invalid Transfer Syntax UID (e.g. JPEG 2000 Lossless) but are actually
    uncompressed.

    Args:
        dcm:
            DICOM object with pixel data
        dims:
            Tuple containing expected image shape

    Returns:
        Numpy ndarray of pixel data
    """
    dtype = np.uint16 if dcm.BitsAllocated == 16 else np.uint8
    return np.frombuffer(dcm.PixelData, dtype=dtype).reshape(dims)


def dcm_to_pixels(dcm: Dicom, dims: Tuple[int, ...], strict_interp: bool) -> ndarray:
    """
    Try to parse pixel data according to PyDicom's default handling,
    and if that fails then try to parse according an alternative method.

    Args:
        dcm:
            DICOM object with pixel data
        dims:
            Tuple containing expected image shape
        strict_interp:
            If true, don't make any assumptions for trying to work around parsing errors

    Returns:
        Numpy ndarray of pixel data
    """
    try:
        return np.ndarray(dims, dcm.pixel_array.dtype, dcm.pixel_array)
    except ValueError as e:
        msg = (
            f"WARNING: (0002, 0010) Transfer Syntax UID does not appear to be correct. PyDicom raised this error: '{e}'"
        )
        if strict_interp:
            raise ValueError(msg)
        else:
            print(msg)
        return uncompressed_dcm_to_pixels(dcm, dims)


def read_dicom_image(
    dcm: Dicom,
    stop_before_pixels: bool = False,
    shape: Optional[Tuple[int, ...]] = None,
    strict_interp: bool = False,
) -> ndarray:
    r"""
    Reads image data from an open DICOM file into a numpy array.

    Args:
        dcm:
            DICOM object to load images from
        stop_before_pixels:
            If true, return randomly generated data
        strict_interp:
            If true, don't make any assumptions for trying to work around parsing errors
        shape:
            Manual shape override when ``stop_before_pixels`` is true. Should not include a channel dimension

    Shape:
        - Output: :math:`(1, H, W)` or :math:`(1, D, H, W)`
    """
    # some dicoms dont have any image data - raise NoImageError
    for necessary_field in ["Rows", "PhotometricInterpretation"]:
        if shape is None and not hasattr(dcm, necessary_field):
            raise NoImageError()

    if shape is None:
        # If NumberOfFrames is 1 or not defined, we treat the DICOM image as a single channel 2D image (i.e. 1xHxW).
        # If NumberOfFrames is greater than 1, we treat the DICOM image as a single channel 3D image (i.e. 1xDxHxW).
        D, H, W = [int(v) for v in [dcm.get("NumberOfFrames", 1), dcm.Rows, dcm.Columns]]
        dims = (1, D, H, W) if D > 1 else (1, H, W)
    else:
        dims = (1,) + shape

    assert dims[0] == 1, "channel dim == 1"
    assert 3 <= len(dims) <= 4, str(dims)

    # return random pixel data in correct shape when stop_before_pixels=True
    if stop_before_pixels:
        return np.random.randint(0, 2 ** 10, dims)

    pixels = dcm_to_pixels(dcm, dims, strict_interp)

    # in some dicoms, pixel value of 0 indicates white
    if is_inverted(dcm.PhotometricInterpretation):  # type: ignore
        pixels = invert_color(pixels)

    # some dicoms have different endianness - convert to native byte order
    if not is_native_byteorder(pixels):
        pixels = pixels.byteswap().newbyteorder()
    assert is_native_byteorder(pixels)

    # numpy byte order needs to explicitly be native "=" for torch conversion
    if pixels.dtype.byteorder != "=":
        pixels = pixels.newbyteorder("=")

    return pixels

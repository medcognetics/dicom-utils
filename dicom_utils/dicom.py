#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from typing import Dict, Final, Iterator, List, Optional, Tuple, Union

import numpy as np
import pydicom
from numpy import ndarray
from pydicom.uid import UID

from .logging import logger
from .types import Dicom, Window
from .volume import KeepVolume, VolumeHandler


# Taken from https://pydicom.github.io/pydicom/dev/old/image_data_handlers.html
TransferSyntaxUIDs: Final[Dict[str, str]] = {
    "1.2.840.10008.1.2.1": "Explicit VR Little Endian",
    "1.2.840.10008.1.2": "Implicit VR Little Endian",
    "1.2.840.10008.1.2.2": "Explicit VR Big Endian",
    "1.2.840.10008.1.2.1.99": "Deflated Explicit VR Little Endian",
    "1.2.840.10008.1.2.5": "RLE Lossless",
    "1.2.840.10008.1.2.4.50": "JPEG Baseline (Process 1)",
    "1.2.840.10008.1.2.4.51": "JPEG Extended (Process 2 and 4)",
    "1.2.840.10008.1.2.4.57": "JPEG Lossless (Process 14)",
    "1.2.840.10008.1.2.4.70": "JPEG Lossless (Process 14, SV1)",
    "1.2.840.10008.1.2.4.80": "JPEG LS Lossless",
    "1.2.840.10008.1.2.4.81": "JPEG LS Lossy",
    "1.2.840.10008.1.2.4.90": "JPEG2000 Lossless",
    "1.2.840.10008.1.2.4.91": "JPEG2000",
    "1.2.840.10008.1.2.4.92": "JPEG2000 Multi-component Lossless",
    "1.2.840.10008.1.2.4.93": "JPEG2000 Multi-component",
}


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


def strict_dcm_to_pixels(dcm: Dicom, dims: Tuple[int, ...]) -> ndarray:
    """
    Interpret pixel data according to the TransferSyntaxUID stored in the DICOM dataset object.

    Args:
        dcm:
            DICOM object with pixel data
        dims:
            Tuple containing expected image shape

    Returns:
        Numpy ndarray of pixel data
    """
    return np.ndarray(dims, dcm.pixel_array.dtype, dcm.pixel_array)


def loose_dcm_to_pixels(dcm: Dicom, dims: Tuple[int, ...]) -> ndarray:
    """
    Try all supported TransferSyntaxUIDs until one succeeds.
    Some mammograms have a mismatch between the TransferSyntaxUID and how the pixel data is actually encoded.

    Args:
        dcm:
            DICOM object with pixel data
        dims:
            Tuple containing expected image shape

    Returns:
        Numpy ndarray of pixel data
    """
    for transfer_syntax_uid in TransferSyntaxUIDs.keys():
        try:
            dcm.file_meta.TransferSyntaxUID = UID(transfer_syntax_uid)
            pixels = strict_dcm_to_pixels(dcm, dims)
            logger.warning(
                f"Able to parse pixels according to '{dcm.file_meta.TransferSyntaxUID}' "
                f"({TransferSyntaxUIDs[dcm.file_meta.TransferSyntaxUID]})"
            )
            return pixels
        except Exception:
            """Don't do anything, just see if the next TransferSyntaxUID works."""
    raise ValueError("Unable to parse the pixel array after trying all possible TransferSyntaxUIDs.")


def dcm_to_pixels(dcm: Dicom, dims: Tuple[int, ...], strict_interp: bool) -> ndarray:
    """
    Try to parse pixel data according to a conformant interpretation,
    and if that fails then try to parse according to an alternative method if strict_interp==False.

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
        return strict_dcm_to_pixels(dcm, dims)
    except Exception as e:
        msg = (
            f"TransferSyntaxUID (0002, 0010) '{dcm.file_meta.TransferSyntaxUID}' "
            f"({TransferSyntaxUIDs[dcm.file_meta.TransferSyntaxUID]}) "
            f"does not appear to be correct. pydicom raised this error: '{e}'"
        )
        if strict_interp:
            raise ValueError(msg)
        logger.warning(msg)
        return loose_dcm_to_pixels(dcm, dims)


def read_dicom_image(
    dcm: Dicom,
    stop_before_pixels: bool = False,
    shape: Optional[Tuple[int, ...]] = None,
    strict_interp: bool = False,
    volume_handler: VolumeHandler = KeepVolume(),
    apply_window: bool = True,
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

    # apply volume handling for 3D data
    if len(dims) == 4:
        dcm = volume_handler(dcm)
        D: int = int(dcm.get("NumberOfFrames", 1))
        dims = (1, D, *dims[-2:]) if D > 1 else (1, *dims[-2:])

    pixels = dcm_to_pixels(dcm, dims, strict_interp)

    # some dicoms have different endianness - convert to native byte order
    if not is_native_byteorder(pixels):
        pixels = pixels.byteswap().newbyteorder()
    assert is_native_byteorder(pixels)

    # numpy byte order needs to explicitly be native "=" for torch conversion
    if pixels.dtype.byteorder != "=":
        pixels = pixels.newbyteorder("=")

    if apply_window:
        window = Window.from_dicom(dcm)
        pixels = window.apply(pixels)
    else:
        pixels = pixels - pixels.min()

    ## in some dicoms, pixel value of 0 indicates white
    if is_inverted(dcm.PhotometricInterpretation):
        pixels = invert_color(pixels)

    return pixels


def path_to_dicom_path_list(path: Path) -> List[Path]:
    if path.is_dir():
        return [f for f in path.iterdir() if has_dicm_prefix(f)]
    if path.is_file() and has_dicm_prefix(path):
        return [path]
    else:
        raise FileNotFoundError(path)


def path_to_dicoms(path: Path) -> Iterator[Dicom]:
    for source in path_to_dicom_path_list(path):
        try:
            yield pydicom.dcmread(source)
        except Exception as e:
            logger.info(e)


def num_pixels(source: Union[Path, Dicom]) -> int:
    if isinstance(source, Path):
        if not source.is_file():
            raise FileNotFoundError(source)
        with pydicom.dcmread(source, stop_before_pixels=True, specific_tags=["Rows", "Columns", "NumberOfFrames"]) as dcm:
            return num_pixels(dcm)

    for necessary_field in ["Rows", "Columns"]:
        if not hasattr(source, necessary_field):
            return 0
    D, H, W = [int(v) for v in [source.get("NumberOfFrames", 1), source.Rows, source.Columns]]
    return D*H*W

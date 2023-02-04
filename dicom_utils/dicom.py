#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from os import PathLike
from pathlib import Path
from time import time
from typing import Callable, Dict, Final, Iterator, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import pydicom
from numpy import ndarray
from pydicom import FileDataset
from pydicom.encaps import encapsulate
from pydicom.pixel_data_handlers.util import apply_voi_lut
from pydicom.uid import UID, ExplicitVRLittleEndian, ImplicitVRLittleEndian

from .basic_offset_table import BasicOffsetTable
from .logging import logger
from .types import Dicom, PhotometricInterpretation
from .volume import KeepVolume, VolumeHandler


try:
    # NOTE: This module will be open-sourced in the future
    import pynvjpeg  # type: ignore
except Exception:
    pynvjpeg = None


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


# Pillow is relatively slow so we want to make sure that other handlers are used instead
default_data_handlers: List[Callable] = pydicom.config.pixel_data_handlers  # type: ignore
data_handlers = [h for h in default_data_handlers if "pillow" not in h.__name__]
pydicom.config.pixel_data_handlers = data_handlers  # type: ignore
assert (a := len(data_handlers) + 1) == (b := len(default_data_handlers)), f"Unexpected data handlers ({a} != {b})"


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
    warn(
        "is_inverted is deprecated. Please use PhotometricInterpretation.is_inverted",
        DeprecationWarning,
    )
    return PhotometricInterpretation.from_str(photo_interp).is_inverted


def invert_color(img: ndarray) -> ndarray:
    """The maximum value will become the minimum and vice versa"""
    return np.max(img) - img


def has_dicm_prefix(filename: Union[str, PathLike]) -> bool:
    """DICOM files have a 128 byte preamble followed by bytes 'DICM'."""
    with open(filename, "rb") as f:
        f.seek(128)
        return f.read(4) == b"DICM"


def is_compressed(dcm: Dicom) -> bool:
    r"""Checks if a DICOM is using a compressed transfer syntax"""
    syntax = dcm.file_meta.TransferSyntaxUID
    return syntax.is_compressed


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
    try:
        pixels = apply_voi_lut(dcm.pixel_array, dcm)
    except Exception:
        pixels = dcm.pixel_array
    return pixels.reshape(dims)


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
    override_shape: Optional[Tuple[int, ...]] = None,
    strict_interp: bool = False,
    volume_handler: VolumeHandler = KeepVolume(),
    as_uint8: bool = False,
    use_nvjpeg: Optional[bool] = False,
    nvjpeg_batch_size: Optional[int] = None,
) -> ndarray:
    r"""
    Reads image data from an open DICOM file into a numpy array.

    Args:
        dcm:
            DICOM object to load images from
        stop_before_pixels:
            If ``True``, return randomly generated data
        shape:
            Manual shape override when ``stop_before_pixels`` is true. Should not include a channel dimension
        strict_interp:
            If ``True``, don't make any assumptions for trying to work around parsing errors
        volume_handler:
            Handler for processing 3D volumes
        as_uint8:
            If ``True``, convert non-uint8 outputs to uint8 using min/max normalization
        use_nvjpeg:
            If ``True``, decompress JPEG2000 images via GPU
        nvjpeg_batch_size:
            Batch size for GPU JPEG2000 decompression

    Shape:
        - Output: :math:`(C, H, W)` or :math:`(C, D, H, W)`
    """
    # some dicoms dont have any image data - raise NoImageError
    for necessary_field in ["Rows", "PhotometricInterpretation"]:
        if override_shape is None and not hasattr(dcm, necessary_field):
            raise NoImageError()
    pm = PhotometricInterpretation.from_dicom(dcm)

    C = pm.num_channels
    if override_shape is None:
        # If NumberOfFrames is 1 or not defined, we treat the DICOM image as a single channel 2D image (i.e. 1xHxW).
        # If NumberOfFrames is greater than 1, we treat the DICOM image as a single channel 3D image (i.e. 1xDxHxW).
        D, H, W = [int(v) for v in [dcm.get("NumberOfFrames", 1), dcm.Rows, dcm.Columns]]
        dims = (C, D, H, W) if D > 1 else (C, H, W)
    else:
        dims = (C,) + override_shape

    assert dims[0] in (1, 3), "channel dim == 1 or 3"
    assert 3 <= len(dims) <= 4, str(dims)

    # return random pixel data in correct shape when stop_before_pixels=True
    if stop_before_pixels or override_shape:
        return np.random.randint(0, 2**10, dims)

    # validation of compressed data
    if is_compressed(dcm) and (num_frames := dcm.get("NumberOfFrames", 1)) > 1:
        if BasicOffsetTable.is_present(dcm):
            bot = BasicOffsetTable.from_stream(dcm)
            if not (bot.is_valid and bot.num_frames == num_frames):
                if strict_interp:
                    raise ValueError(
                        "DICOM does not appear to have a valid basic offset table:\n"
                        f"NumberOfFrames: {dcm.get('NumberOfFrames', 1)}\n"
                        f"Offsets: {list(bot)}"
                    )
                else:
                    # replace the invalid BOT with an empty one
                    dcm.PixelData = BasicOffsetTable.remove_from(dcm)
                    bot = BasicOffsetTable.default()
                    dcm.PixelData = bot.prepend_to(dcm.PixelData)
            assert bot.is_valid

    # apply volume handling for 3D data
    if len(dims) == 4:
        dcm = volume_handler(dcm)
        D: int = int(dcm.get("NumberOfFrames", 1))
        dims = (C, D, *dims[-2:]) if D > 1 else (C, *dims[-2:])

    # decompress with GPU if requested
    # TODO If the volume handler is ReduceVolume, we should be decompressing
    # before the handler is applied instead of after.
    # But for every other handler we should decompress after.
    # We need to figure out the best way to deal with this.
    if use_nvjpeg is None or use_nvjpeg:
        dcm = decompress(dcm, use_nvjpeg=use_nvjpeg, batch_size=nvjpeg_batch_size)

    # DICOM is channels last, so permute dims
    channels_last_dims = *dims[1:], dims[0]
    pixels = dcm_to_pixels(dcm, channels_last_dims, strict_interp)
    pixels = np.moveaxis(pixels, -1, 0)
    assert tuple(pixels.shape) == dims

    # some dicoms have different endianness - convert to native byte order
    if not is_native_byteorder(pixels):
        pixels = pixels.byteswap().newbyteorder()
    assert is_native_byteorder(pixels)

    # numpy byte order needs to explicitly be native "=" for torch conversion
    if pixels.dtype.byteorder != "=":
        pixels = pixels.newbyteorder("=")

    # in some dicoms, pixel value of 0 indicates white
    if pm.is_inverted:
        pixels = invert_color(pixels)

    # convert to uint8 if requested
    if as_uint8 and pixels.dtype != np.uint8:
        max, min = pixels.max(), pixels.min()
        delta = (max - min).clip(min=1)
        pixels = (pixels - min) / delta
        pixels = (pixels * 255).astype(np.uint8)

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


def set_pixels(dcm: FileDataset, arr: np.ndarray, syntax: UID) -> FileDataset:
    r"""Sets the pixels of a DICOM object from a numpy array, accounting for TransferSyntaxUID."""
    if syntax.is_compressed:
        # NOTE: dcm.file_meta.TransferSyntaxUID must be an uncompressed TSUID when calling dcm.compress().
        # swap in a temporary uncompressed TSUID to address this
        dcm.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        # encapsulate frames
        if int(dcm.NumberOfFrames) > 1:
            new_data = encapsulate([a.tobytes() for a in arr], has_bot=False)
        else:
            new_data = encapsulate([arr.tobytes()])
        # update dcm with compressed attributes
        dcm.PixelData = new_data
        dcm.compress(syntax)
        dcm.file_meta.TransferSyntaxUID = syntax
    else:
        dcm.PixelData = arr.tobytes()
    dcm.file_meta.TransferSyntaxUID = syntax
    return dcm


def decompress(
    dcm: Dicom,
    strict: bool = False,
    use_nvjpeg: Optional[bool] = None,
    batch_size: Optional[int] = None,
    verbose: bool = False,
) -> Dicom:
    r"""Decompress pixel data of a DICOM object

    Args:
        dcm: DICOM object to decompress
        strict: Raise an error if a decompressed DICOM is given as input
        use_nvjpeg: Use GPU accelerated decompression. Set to ``None`` to use if available
        batch_size: GPU decompression batch size. Set to ``None`` to defer to environment variable
        ``NVJPEG2K_BATCH_SIZE``

    Returns:
        DICOM object with decompressed pixel data and ExplicitVRLittleEndian transfer syntax.
    """
    tsuid = dcm.file_meta.TransferSyntaxUID
    if not tsuid.is_compressed:
        if strict:
            raise RuntimeError(f"TransferSyntaxUID {tsuid} is already decompressed")
        else:
            return dcm

    use_nvjpeg = use_nvjpeg if use_nvjpeg is not None else nvjpeg2k_is_available()
    if use_nvjpeg:
        batch_size = batch_size or int(os.environ.get("NVJPEG2K_BATCH_SIZE", 1))
        pixels = nvjpeg_decompress(dcm, batch_size, verbose)
    else:
        pixels = dcm.pixel_array
    assert isinstance(dcm, FileDataset)
    dcm = set_pixels(dcm, pixels, ExplicitVRLittleEndian)
    return dcm


def nvjpeg2k_is_available() -> bool:  # pragma: no cover
    return pynvjpeg is not None


# TODO: support num_frames % batch_size != 0 in C++ extension
def _nvjpeg_get_batch_size(batch_size: int, num_frames: int) -> int:  # pragma: no cover
    while num_frames % batch_size != 0 and batch_size > 1:
        batch_size = batch_size - 1
    assert batch_size >= 1
    assert num_frames % batch_size == 0
    return batch_size


def nvjpeg_decompress(
    dcm: Dicom,
    batch_size: int = 4,
    verbose: bool = False,
) -> np.ndarray:  # pragma: no cover
    r"""Decompress pixel data of a DICOM object with NVJPEG2000

    Args:
        dcm: DICOM object to decompress
        batch_size: GPU decompression batch size. Set to ``None`` to defer to environment variable
        ``NVJPEG2K_BATCH_SIZE``.

    Returns:
        Decompressed pixel array
    """
    if not nvjpeg2k_is_available():
        raise ImportError("pynvjpeg is not available")

    num_frames = int(dcm.get("NumberOfFrames", 1))
    rows = dcm.Rows
    cols = dcm.Columns

    batch_size = _nvjpeg_get_batch_size(batch_size, num_frames)
    pixels = dcm.PixelData
    assert pynvjpeg is not None  # To fix a type error on the next line even though we already checked for this
    result = pynvjpeg.decode_frames_jpeg2k(pixels, len(pixels), rows, cols, batch_size)
    return result

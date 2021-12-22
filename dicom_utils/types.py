#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, cast

import numpy as np
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence


IMAGE_TYPE = 0x00080008
WINDOW_CENTER = 0x00281050
WINDOW_WIDTH = 0x00281051

Dicom = Dataset
DicomAttributeSequence = Sequence


class SimpleImageType(Enum):
    UNKNOWN = 0
    NORMAL = 1
    SVIEW = 2
    TOMO = 3

    @staticmethod
    def from_dicom(dcm: Dicom) -> "SimpleImageType":
        img_type = ImageType.from_dicom(dcm)
        return img_type.to_simple_image_type()

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        if self is self.UNKNOWN:
            return "unknown"
        elif self is self.NORMAL:
            return "2d"
        elif self is self.SVIEW:
            return "s-view"
        elif self is self.TOMO:
            return "tomo"
        else:
            raise RuntimeError("unknown ImageType value")


@dataclass(frozen=True)
class ImageType:
    """Container for DICOM metadata fields related to the image type.

    Contains the following attributes:

        * ``"pixels"`` - First element of the IMAGE_TYPE field.
        * ``"exam"`` - Second element of the IMAGE_TYPE field.
        * ``"flavor"`` - Third element of the IMAGE_TYPE field.
        * ``"extras"`` - Additional IMAGE_TYPE fields if available
        * ``"NumberOfFrames"`` - Frame count (for TOMO images only)
        * ``"model"`` - Manufacturer's model name
    """

    pixels: str
    exam: str
    flavor: Optional[str] = None
    extras: Optional[List[str]] = None
    NumberOfFrames: Optional[int] = None
    model: Optional[str] = None

    def __bool__(self) -> bool:
        return bool(self.pixels) and bool(self.exam)

    def __post_init__(self):
        if (nf := self.NumberOfFrames) is not None and not isinstance(nf, int):
            object.__setattr__(self, "NumberOfFrames", int(nf))
        assert self.NumberOfFrames is None or self.NumberOfFrames > 0

    def simple_repr(self) -> str:
        s = "|".join(x for x in (self.pixels, self.exam))
        if self.flavor is not None:
            s += f"|{self.flavor}" if self.flavor else "|''"
        for elem in self.extras or []:
            if not elem or elem.isdigit():
                continue
            s += f"|{elem}"
        return s

    @classmethod
    def from_dicom(cls, dcm: Dicom) -> "ImageType":
        result: Dict[str, Any] = {}
        result["NumberOfFrames"] = dcm.get("NumberOfFrames", None)
        result["model"] = dcm.get("ManufacturerModelName", None)

        if IMAGE_TYPE not in dcm.keys():
            return cls("", "", **result)

        # fields 1 and 2 should always be present
        image_type = cast(List[str], dcm[IMAGE_TYPE].value)
        pixels, exam = image_type[:2]
        result["pixels"] = pixels
        result["exam"] = exam

        # there might be a field 3
        if len(image_type) >= 3:
            flavor = image_type[2]
            result["flavor"] = flavor

        if len(image_type) >= 4:
            result["extras"] = image_type[3:]

        assert "pixels" in result.keys()
        assert "exam" in result.keys()
        return cls(**result)

    def to_simple_image_type(self) -> SimpleImageType:
        r"""Converts the :class:`ImageType` container into a :class:`SimpleImageType`.

        .. warning:
            This method may be unreliable in its detection of S-View images
        """
        # if fields 1 and 2 were missing, we know nothing
        if not self.pixels and self.exam:
            return SimpleImageType.UNKNOWN

        pixels = self.pixels.lower()
        exam = self.exam.lower()
        flavor = (self.flavor or "").lower()
        extras = self.extras
        num_frames = self.NumberOfFrames or 1
        machine = (self.model or "").lower()

        # very solid rules
        if num_frames > 1:
            return SimpleImageType.TOMO
        if "original" in pixels:
            return SimpleImageType.NORMAL

        # ok rules
        if extras is not None and any("generated_2d" in x.lower() for x in extras):
            return SimpleImageType.SVIEW

        # not good rules
        if pixels == "derived" and exam == "primary" and machine == "fdr-3000aws" and flavor != "post_contrast":
            return SimpleImageType.SVIEW

        return SimpleImageType.NORMAL


@dataclass
class Window:
    r"""Contains data related to the window of pixel intensities for display.
    Window levels are defined realtive to the PhotometricInterpretation of the DICOM image.
    As such, application of windows should be done before application of inversions.
    """
    center: int
    width: int

    # unused, should eventually store window explanation string
    descriptor: Optional[str] = None

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}("
        s += f"center={self.center}, "
        s += f"width={self.width}, "
        s += f"low={self.lower_bound}, "
        s += f"high={self.upper_bound}"
        s += ")"
        return s

    @classmethod
    def from_dicom(cls, dcm: Dicom) -> "Window":
        center = dcm.get(WINDOW_CENTER, None)
        width = dcm.get(WINDOW_WIDTH, None)

        # fallback to pixel data if center or width is missing
        if center is None or width is None:
            pixels = dcm.pixel_array
            max, min = pixels.max(), pixels.min()
            center = (max - min) // 2 + min
            width = max - min

        # for single window levels
        elif isinstance(center.value, (str, float)):
            center = center.value
            width = width.value

        # if multiple levels, read the first
        # TODO allow window selection by name
        else:
            center = center.value[0]
            width = width.value[0]

        center = int(center)  # type: ignore
        width = int(width)  # type: ignore
        return cls(center, width)

    @property
    def constrained_width(self) -> int:
        r"""The window width after being constrained to non-negative values"""
        return int(self.upper_bound - self.lower_bound)

    @property
    def lower_bound(self) -> float:
        return max(self.center - (self.width // 2), 0)

    @property
    def upper_bound(self) -> float:
        return self.center + (self.width // 2)

    def apply(self, pixels: np.ndarray) -> np.ndarray:
        r"""Applies this window to an array of pixel data. Since DICOM windows are defined
        relative to the original pixel values, windows should be applied only to original
        pixel values.
        """
        # record dtype so we can restore floats to input dtype
        pixels = pixels.clip(min=self.lower_bound, max=self.upper_bound)
        pixels = pixels - self.lower_bound
        return pixels


__all__ = ["Dicom", "ImageType", "Window"]

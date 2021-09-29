#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, cast

from pydicom.dataset import Dataset
from pydicom.sequence import Sequence


IMAGE_TYPE = 0x00080008

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


__all__ = ["Dicom", "ImageType"]

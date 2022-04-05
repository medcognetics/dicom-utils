#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Final, Iterable, List, Optional, TypeVar, cast

import numpy as np
from pydicom import DataElement
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from .tags import Tag


IMAGE_TYPE = 0x00080008
WINDOW_CENTER = 0x00281050
WINDOW_WIDTH = 0x00281051
UNKNOWN: Final = -1

Dicom = Dataset
DicomAttributeSequence = Sequence


def get_tag_values(tags: Iterable[Tag], dcm: Dicom) -> Dict[Tag, Any]:
    tags = {tag: dcm[tag].value if tag in dcm else None for tag in tags}
    return tags


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

    @classmethod
    def from_str(cls, s: str) -> "SimpleImageType":
        if "tomo" in s:
            return cls.TOMO
        elif "view" in s or "synth" in s:
            return cls.SVIEW
        elif "2d" in s or "ffdm" in s:
            return cls.NORMAL
        else:
            return cls.UNKNOWN


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
        * ``"series_description"``
    """

    pixels: str
    exam: str
    flavor: Optional[str] = None
    extras: Optional[List[str]] = None
    NumberOfFrames: Optional[int] = None
    model: Optional[str] = None
    series_description: Optional[str] = None

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
        result["series_description"] = dcm.get("SeriesDescription", None)

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
        series_description = (self.series_description or "").lower()

        # very solid rules
        if num_frames > 1:
            return SimpleImageType.TOMO
        if series_description and ("s-view" in series_description or "c-view" in series_description):
            return SimpleImageType.SVIEW
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


class PhotometricInterpretation(Enum):
    r"""Enumeration of PhotometricInterpretation values under the DICOM
    standard. Values pulled from:

        https://dicom.innolitics.com/ciods/rt-dose/image-pixel/00280004
    """
    UNKNOWN = 0
    MONOCHROME1 = 1
    MONOCHROME2 = 2
    PALETTE_COLOR = auto()
    RGB = auto()
    HSV = auto()
    ARGB = auto()
    CMYK = auto()
    YBR_FULL = auto()
    YBR_FULL_422 = auto()
    YBR_PARTIAL_422 = auto()
    YBR_PARTIAL_420 = auto()
    YBR_ICT = auto()
    YBR_RCT = auto()

    def __bool__(self) -> bool:
        return self != PhotometricInterpretation.UNKNOWN

    @property
    def is_monochrome(self) -> bool:
        return 1 <= self.value <= 2

    @property
    def num_channels(self) -> int:
        return 1 if self.is_monochrome else 3

    @property
    def is_inverted(self) -> bool:
        return self == PhotometricInterpretation.MONOCHROME1

    @classmethod
    def from_str(cls, string: str) -> "PhotometricInterpretation":
        string = string.upper()
        return getattr(cls, string, PhotometricInterpretation.UNKNOWN)

    @classmethod
    def from_dicom(cls, dcm: Dicom) -> "PhotometricInterpretation":
        val = dcm.get(Tag.PhotometricInterpretation, None)
        return PhotometricInterpretation.UNKNOWN if val is None else cls.from_str(val.value)


T = TypeVar("T")


class EnumMixin(Enum):
    def __bool__(self) -> bool:
        return not self.is_unknown

    def __repr__(self) -> str:
        name = self.simple_name
        return f"{self.__class__.__name__}({name})"

    def __add__(self: T, other: T) -> T:
        return self or other

    def __mul__(self: T, other: T) -> T:
        return self or other

    @property
    def is_unknown(self) -> bool:
        return self.value == UNKNOWN

    @property
    def simple_name(self) -> str:
        return self.name.lower().replace("_", " ")

    @staticmethod
    def get_required_tags() -> List[Tag]:
        raise NotImplementedError("get_required_tags() has not been implemented for this class")


class Laterality(EnumMixin):
    UNKNOWN = UNKNOWN

    NONE = 0
    LEFT = 1
    RIGHT = 2
    BILATERAL = 3

    @staticmethod
    def get_required_tags() -> List[Tag]:
        return [Tag.ImageLaterality, Tag.FrameLaterality, Tag.SharedFunctionalGroupsSequence]

    @classmethod
    def from_str(cls, string: str) -> "Laterality":
        string = string.lower()
        if string == "none":
            return cls.NONE
        if "bi" in string:  # TODO what other patterns describe bilateral?
            return cls.BILATERAL
        if "r" in string or "d" in string:
            return cls.RIGHT
        if "l" in string or "e" in string:
            return cls.LEFT
        return cls.UNKNOWN

    @classmethod
    def from_tags(cls, tags: Dict[int, Any]) -> "Laterality":
        # Take subset of 'tags' so that unit tests will fail if we don't maintain get_required_tags()
        tags = {k: v for k, v in tags.items() if k in cls.get_required_tags()}

        # first try reading ImageLaterality
        laterality = tags.get(Tag.ImageLaterality, None)

        # fall back to Tag.FrameLaterality
        if laterality is None:
            try:
                laterality = (
                    tags.get(Tag.SharedFunctionalGroupsSequence)[0]
                    .get(Tag.FrameAnatomySequence)
                    .value[0]
                    .get(Tag.FrameLaterality)
                    .value
                )
            except Exception:
                pass

        if isinstance(laterality, str):
            laterality = laterality.strip().lower()
            if laterality == "l":
                return cls.LEFT
            if laterality == "r":
                return cls.RIGHT
        # TODO is there a DICOM value for bilateral?
        return cls.UNKNOWN

    @classmethod
    def from_dicom(cls, dcm: Dicom) -> "Laterality":
        return cls.from_tags({int(tag): value for tag, value in get_tag_values(cls.get_required_tags(), dcm).items()})

    @classmethod
    def from_case_notes(cls, notes: str) -> "Laterality":
        notes = notes.lower()
        if "left" in notes:
            return cls.LEFT
        elif "right" in notes:
            return cls.RIGHT
        elif "bilateral" in notes:
            return cls.BILATERAL
        else:
            return cls.UNKNOWN

    @property
    def short_str(self) -> str:
        if self == Laterality.LEFT:
            return "l"
        elif self == Laterality.RIGHT:
            return "r"
        return ""


class ViewPosition(EnumMixin):
    UNKNOWN = UNKNOWN

    CC = 1
    MLO = 2
    ML = 3

    @staticmethod
    def get_required_tags() -> List[Tag]:
        return [Tag.ViewPosition, Tag.ViewCodeSequence]

    @classmethod
    def from_str(cls, string: str) -> "ViewPosition":
        string = string.lower()
        if "cc" in string:
            return cls.CC
        if "mlo" in string:
            return cls.MLO
        if "ml" in string:
            return cls.ML
        return cls.UNKNOWN

    @classmethod
    def from_tags(cls, tags: Dict[int, Any]) -> "ViewPosition":
        # Take subset of 'tags' so that unit tests will fail if we don't maintain get_required_tags()
        tags = {k: v for k, v in tags.items() if k in cls.get_required_tags()}
        view_position = cls.from_view_position_tag(tags.get(Tag.ViewPosition, None))
        return (
            view_position
            if view_position is not cls.UNKNOWN
            else cls.from_view_code_sequence_tag(tags.get(Tag.ViewCodeSequence, None))
        )

    @classmethod
    def from_view_position_tag(cls, view_position: Optional[str]) -> "ViewPosition":
        if isinstance(view_position, str):
            view_position = view_position.strip().lower()
            if view_position == "mlo":
                return cls.MLO
            elif view_position == "cc":
                return cls.CC
            elif view_position == "ml":
                return cls.ML
        return cls.UNKNOWN

    @classmethod
    def from_view_code_sequence_tag(cls, view_code_sequence: Optional[DataElement]) -> "ViewPosition":
        for view_code in view_code_sequence or []:
            meaning = view_code.get("CodeMeaning", None)
            if isinstance(meaning, str):
                meaning = meaning.strip().lower()
                if meaning == "cranio-caudal":
                    return cls.CC
                elif meaning == "medio-lateral oblique":
                    return cls.MLO
                elif meaning == "medio-lateral":
                    return cls.ML
        return cls.UNKNOWN

    @classmethod
    def from_dicom(cls, dcm: Dicom) -> "ViewPosition":
        return cls.from_tags({int(tag): value for tag, value in get_tag_values(cls.get_required_tags(), dcm).items()})

    @property
    def short_str(self) -> str:
        if self == ViewPosition.CC:
            return "cc"
        elif self == ViewPosition.MLO:
            return "mlo"
        elif self == ViewPosition.ML:
            return "ml"
        return ""


__all__ = ["Dicom", "ImageType", "Window", "PhotometricInterpretation", "EnumMixin", "Laterality", "ViewPosition"]

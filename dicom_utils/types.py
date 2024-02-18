#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Final, Iterable, Iterator, List, NamedTuple, Optional, TypeVar, Union, cast

from pydicom import DataElement
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence

from .tags import Tag


UNKNOWN: Final = -1


Dicom = Dataset
DicomAttributeSequence = Sequence


class DicomValueError(ValueError):
    pass


class DicomKeyError(ValueError):
    pass


def get_value(dcm: Dicom, tag: Tag, default: Any, try_file_meta: bool = False) -> Any:
    if tag in dcm:
        return dcm.get(tag).value
    elif try_file_meta and hasattr(dcm, "file_meta") and tag in dcm.file_meta:
        return dcm.file_meta.get(tag).value
    return default


def get_tag_values(tags: Iterable[Tag], dcm: Dicom) -> Dict[Tag, Any]:
    tags = {tag: get_value(dcm, tag, None) for tag in tags}
    return tags


# TODO: these would be better in dicom.py, but circular import issues
def iterate_view_codes(dcm: Dataset) -> Iterator[Dataset]:
    r"""Iterates over all view codes in an input."""
    view_code_sequence = (get_value(dcm, Tag.ViewCodeSequence, []) or []) if isinstance(dcm, Dataset) else dcm
    for view_code in view_code_sequence:
        yield view_code


def iterate_view_modifier_codes(dcm: Dataset) -> Iterator[Dataset]:
    r"""Iterates over all view modifier codes in an input."""
    # view modifier code can be at top level of dicom, or nested in view codes
    modifier_sequence = (get_value(dcm, Tag.ViewModifierCodeSequence, []) or []) if isinstance(dcm, Dataset) else dcm
    for modifier in modifier_sequence:
        yield modifier
    for view_code in iterate_view_codes(dcm):
        for modifier in iterate_view_modifier_codes(view_code):
            yield modifier


def iterate_shared_functional_groups(dcm: Dataset) -> Iterator[Dataset]:
    r"""Iterates over all view modifier codes in an input."""
    # view modifier code can be at top level of dicom, or nested in view codes
    functional_group_seq = (
        (get_value(dcm, Tag.SharedFunctionalGroupsSequence, []) or []) if isinstance(dcm, Dataset) else dcm
    )
    for seq in functional_group_seq:
        yield seq


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


class MammogramType(EnumMixin):
    r"""Enum over the subcategories of mammograms.

    Supports the following ordering: ``TOMO < FFDM < SYNTH < SFM < UNKNOWN``
    """

    UNKNOWN = 0
    TOMO = 1
    FFDM = 2
    SYNTH = 3
    SFM = 4

    def __lt__(self, other: "MammogramType") -> bool:
        return self.is_preferred_to(other)

    def __le__(self, other: "MammogramType") -> bool:
        return self.is_preferred_to(other) or self == other

    def __bool__(self) -> bool:
        return self != MammogramType.UNKNOWN

    # TODO: MammogramType ordering uses the convention that x < y means x is more preferred than y
    # We may want to change this to x > y means x is more preferred than y. This would be more intuitive
    # but would mean `sorted(vals)` would return the values in the opposite order.
    def is_preferred_to(self, other: "MammogramType") -> bool:
        r"""Returns whether the current mammogram type is preferred to another.

        Args:
            other: The other mammogram type.

        Returns:
            Whether the current mammogram type is preferred to the other.
        """
        if self.is_unknown:
            return False
        elif other.is_unknown:
            return True
        return self.value < other.value

    @staticmethod
    def get_best(types: List["MammogramType"]) -> "MammogramType":
        r"""Returns the best mammogram type from a list of types."""
        if not types:
            raise ValueError("types must not be empty")
        return min(types)

    @property
    def is_unknown(self) -> bool:
        return not bool(self)

    @staticmethod
    def get_required_tags() -> List[Tag]:
        return [Tag.ImageType, Tag.Modality, Tag.NumberOfFrames, Tag.SeriesDescription]

    @staticmethod
    def from_dicom(dcm: Dicom, is_sfm: bool = False, ignore_modality: bool = False) -> "MammogramType":
        if (modality := get_value(dcm, Tag.Modality, None)) not in (None, "MG") and not ignore_modality:
            raise DicomValueError(f"Expected modality=MG, found {modality}")

        # if DICOM is a 3D volume, it must be tomo
        if dcm.get("NumberOfFrames", 1) > 1:
            return MammogramType.TOMO

        img_type = ImageType.from_dicom(dcm)
        pixels = img_type.pixels.lower()
        exam = img_type.exam.lower()
        flavor = (img_type.flavor or "").lower()
        extras = img_type.extras
        machine = get_value(dcm, Tag.ManufacturerModelName, "").lower()
        series_description = get_value(dcm, Tag.SeriesDescription, "").lower()

        # if fields 1 and 2 were missing, we know nothing
        if not img_type.pixels and img_type.exam:
            return MammogramType.UNKNOWN

        # very solid rules
        if is_sfm:
            return MammogramType.SFM
        if series_description and ("s-view" in series_description or "c-view" in series_description):
            return MammogramType.SYNTH
        if "original" in pixels:
            return MammogramType.FFDM

        # ok rules
        if extras is not None and any("generated_2d" in x.lower() for x in extras):
            return MammogramType.SYNTH

        # not good rules
        if pixels == "derived" and exam == "primary" and machine == "fdr-3000aws" and flavor != "post_contrast":
            return MammogramType.SYNTH

        return MammogramType.FFDM

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        if self is self.UNKNOWN:
            return "unknown"
        elif self is self.FFDM:
            return "ffdm"
        elif self is self.SFM:
            return "sfm"
        elif self is self.SYNTH:
            return "s-view"
        elif self is self.TOMO:
            return "tomo"
        else:
            raise RuntimeError("unknown ImageType value")

    @classmethod
    def from_str(cls, s: str) -> "MammogramType":
        if "tomo" in s:
            return cls.TOMO
        elif "view" in s or "synth" in s:
            return cls.SYNTH
        elif "2d" in s or "ffdm" in s:
            return cls.FFDM
        elif "sfm" in s:
            return cls.SFM
        else:
            return cls.UNKNOWN


@dataclass(frozen=True)
class ImageType:
    """Container for DICOM metadata fields related to the image type.

    Contains the following attributes:

        * ``"pixels"`` - First element of the ImageType field.
        * ``"exam"`` - Second element of the ImageType field.
        * ``"flavor"`` - Third element of the ImageType field.
        * ``"extras"`` - Additional ImageType fields if available
    """

    pixels: str
    exam: str
    flavor: Optional[str] = None
    extras: Optional[List[str]] = None

    def __bool__(self) -> bool:
        return bool(self.pixels) and bool(self.exam)

    def __contains__(self, val: Any) -> bool:
        return val == self.pixels or val == self.exam or val == self.flavor or val in (self.extras or [])

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

        if Tag.ImageType not in dcm.keys():
            return cls("", "", **result)

        # fields 1 and 2 should always be present
        image_type = cast(List[str], dcm[Tag.ImageType].value)
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


class Laterality(EnumMixin):
    UNKNOWN = UNKNOWN

    NONE = 0
    LEFT = 1
    RIGHT = 2
    BILATERAL = 3

    @property
    def opposite(self) -> "Laterality":
        if self == Laterality.LEFT:
            return Laterality.RIGHT
        elif self == Laterality.RIGHT:
            return Laterality.LEFT
        return Laterality.UNKNOWN

    @property
    def is_unilateral(self) -> bool:
        return self in (Laterality.LEFT, Laterality.RIGHT)

    @property
    def is_unknown_or_none(self) -> bool:
        return self in (Laterality.UNKNOWN, Laterality.NONE)

    @staticmethod
    def get_required_tags() -> List[Tag]:
        return [
            Tag.Laterality,
            Tag.ImageLaterality,
            Tag.FrameLaterality,
            Tag.SharedFunctionalGroupsSequence,
            Tag.PatientOrientation,
        ]

    @classmethod
    def from_str(cls, string: str) -> "Laterality":
        string = string.strip().lower()
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
        laterality = tags.get(Tag.ImageLaterality, "")

        # next try reading Laterality
        laterality = laterality or tags.get(Tag.Laterality, "")

        # fall back to Tag.FrameLaterality
        if not laterality:
            try:
                laterality = (
                    tags.get(Tag.SharedFunctionalGroupsSequence)[0]  # type: ignore
                    .get(Tag.FrameAnatomySequence)
                    .value[0]
                    .get(Tag.FrameLaterality)
                    .value
                )
            except Exception:
                pass

        # TODO is there a DICOM value for bilateral?
        laterality = laterality.strip().lower() if laterality else ""
        if laterality == "l":
            return cls.LEFT
        elif laterality == "r":
            return cls.RIGHT
        else:
            return cls.UNKNOWN

    @classmethod
    def from_patient_orientation(cls, value: Union[str, List[str]]) -> "Laterality":
        if isinstance(value, str):
            return cls.from_str(value)

        for item in value:
            parts = set(str(item))
            if "L" in parts:
                return cls.RIGHT
            elif "R" in parts:
                return cls.LEFT
            elif result := cls.from_str(item):
                return result
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

    def __add__(self, other: "Laterality") -> "Laterality":
        r"""Reduces two :class:`Laterality` inputs according to the following rules:
        * ANY + BILATERAL -> BILATERAL
        * LEFT + RIGHT -> BILATERAL
        * LEFT + (UNKNOWN/NONE) -> LEFT
        * RIGHT + (UNKNOWN/NONE) -> RIGHT
        * NONE + NONE -> NONE
        * UNKOWN + UNKOWN -> UNKNOWN
        """
        # either input is unknown, fall back to parent __add__
        if self.is_unknown or other.is_unknown:
            return cast(Laterality, super().__add__(other))

        # if either input is bilateral, output should be bilateral
        elif self == Laterality.BILATERAL or other == Laterality.BILATERAL:
            return Laterality.BILATERAL

        # if both inputs are unilateral complements, reduce to bilateral
        # if either inputs is a unilateral non-complements, return the unilateral laterality
        elif self.is_unilateral or other.is_unilateral:
            return (
                Laterality.BILATERAL
                if self.is_unilateral and other.is_unilateral and self != other
                else self if self.is_unilateral else other
            )

        # only remaining possibility
        return Laterality.NONE

    def reduce(self, *other: "Laterality") -> "Laterality":
        return sum((self, *other), Laterality.UNKNOWN)


CC_STRINGS: Final = {"cranio-caudal", "caudal-cranial"}
ML_STRINGS: Final = {"medio-lateral", "medial-lateral"}
LM_STRINGS: Final = {"latero-medial", "lateral-medial"}
MLO_STRINGS: Final = {pattern for s in ML_STRINGS for pattern in (f"{s} oblique", f"oblique {s}")}
LMO_STRINGS: Final = {pattern for s in LM_STRINGS for pattern in (f"{s} oblique", f"oblique {s}")}
XCCL_STRINGS: Final = {s + " exaggerated laterally" for s in CC_STRINGS}
XCCM_STRINGS: Final = {s + " exaggerated medially" for s in CC_STRINGS}
AT_STRINGS: Final = {"axillary tail"}
CV_STRINGS: Final = {"cleavage view", "valley-view"}


class ViewPosition(EnumMixin):
    UNKNOWN = UNKNOWN

    XCCL = auto()
    XCCM = auto()
    CC = auto()
    MLO = auto()
    ML = auto()
    LMO = auto()
    LM = auto()
    AT = auto()
    CV = auto()

    @staticmethod
    def get_required_tags() -> List[Tag]:
        return [Tag.ViewPosition, Tag.ViewCodeSequence]

    @classmethod
    def from_str(cls, string: str, strict: bool = False) -> "ViewPosition":
        string = string.strip().lower()

        # first try strict patterns
        strict_mapping = {
            cls.CC: CC_STRINGS,
            cls.MLO: MLO_STRINGS,
            cls.ML: ML_STRINGS,
            cls.LM: LM_STRINGS,
            cls.LMO: LMO_STRINGS,
            cls.XCCL: XCCL_STRINGS,
            cls.XCCM: XCCM_STRINGS,
            cls.AT: AT_STRINGS,
        }
        for value, keywords in strict_mapping.items():
            if string in keywords:
                return value
            elif value.simple_name == string:
                return value

        if strict:
            return cls.UNKNOWN

        # loose patterns
        for choice in cls:
            keyword = choice.simple_name
            if keyword and keyword in string:
                return choice
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
            return cls.from_str(view_position)
        return cls.UNKNOWN

    @classmethod
    def from_view_code_sequence_tag(cls, view_code_sequence: Optional[DataElement]) -> "ViewPosition":
        for view_code in view_code_sequence or []:
            meaning = view_code.get("CodeMeaning", None)
            if isinstance(meaning, str):
                return cls.from_str(meaning, strict=True)
        return cls.UNKNOWN

    @classmethod
    def from_patient_orientation(cls, value: Union[str, List[str]]) -> "ViewPosition":
        if isinstance(value, str):
            return cls.from_str(value)

        for item in value:
            if item in ("FL", "FR"):
                return cls.MLO
            elif item in ("R", "L"):
                return cls.CC
            elif result := cls.from_str(item):
                return result
        return cls.UNKNOWN

    @classmethod
    def from_dicom(cls, dcm: Dicom) -> "ViewPosition":
        def from_code(code: Dataset) -> ViewPosition:
            return cls.from_str(get_value(code, Tag.CodeMeaning, ""), strict=True)

        view_position = cls.from_str(get_value(dcm, Tag.ViewPosition, ""))
        view_codes = {result for code in iterate_view_codes(dcm) if (result := from_code(code))}
        view_modifier_codes = {result for code in iterate_view_modifier_codes(dcm) if (result := from_code(code))}
        candidates = view_codes.union(view_modifier_codes).union({view_position})
        return sorted(candidates, key=lambda x: x.value)[-1]

    @property
    def short_str(self) -> str:
        if self == self.__class__.UNKNOWN:
            return ""
        return self.name.lower()

    @property
    def is_standard_view(self) -> bool:
        return self in {self.CC, self.MLO}

    @property
    def is_mlo_like(self) -> bool:
        return self in {self.MLO, self.ML, self.LMO, self.LM}

    @property
    def is_cc_like(self) -> bool:
        return self in {self.CC, self.XCCL, self.XCCM}


class MammogramView(NamedTuple):
    laterality: Laterality = Laterality.UNKNOWN
    view: ViewPosition = ViewPosition.UNKNOWN

    @classmethod
    def create(cls, laterality: Optional[Laterality] = None, view: Optional[ViewPosition] = None) -> "MammogramView":
        return cls(
            laterality or Laterality.UNKNOWN,
            view or ViewPosition.UNKNOWN,
        )

    @classmethod
    def from_dicom(cls, dcm: Dicom) -> "MammogramView":
        return cls.create(
            Laterality.from_dicom(dcm),
            ViewPosition.from_dicom(dcm),
        )

    @property
    def is_standard_mammo_view(self) -> bool:
        r"""Checks if this record corresponds to a standard mammography view.
        Standard mammography views are the MLO and CC views.
        """
        return self.view.is_standard_view

    @staticmethod
    def get_required_tags() -> List[Tag]:
        return [*Laterality.get_required_tags(), *ViewPosition.get_required_tags()]

    @property
    def is_mlo_like(self) -> bool:
        return self.view.is_mlo_like

    @property
    def is_cc_like(self) -> bool:
        return self.view.is_cc_like


# Matches floats, incluidng exponential notation
FLOAT_PATTERN = r"\d+\.?\d*(?:[e\-\d]+)?"

PIXEL_SPACING_RE = re.compile(rf"({FLOAT_PATTERN})[^\d.]+({FLOAT_PATTERN})")


@dataclass(frozen=True)
class PixelSpacing:
    r"""Represents detector pixel spacing in mm."""

    row: float
    col: float

    @classmethod
    def from_str(cls, string: str) -> "PixelSpacing":
        # value will be of from [row, col] in mm
        match = PIXEL_SPACING_RE.search(string)
        if match:
            try:
                values = tuple(float(x) for x in match.groups())
                row, col = values
                return cls(row, col)
            except ValueError:
                pass
        raise ValueError(f"Failed to parse PixelSpacing from {string}")

    @classmethod
    def from_tags(cls, tags: Dict[Tag, Any]) -> "PixelSpacing":
        for tag, spacing in tags.items():
            try:
                if isinstance(spacing, str):
                    return cls.from_str(spacing)
                elif isinstance(spacing, MultiValue):
                    row, col = tuple(float(x) for x in spacing)
                    return cls(row, col)
            except ValueError:
                pass
        raise RuntimeError("Failed to create PixelSpacing from DICOM")

    @classmethod
    def from_dicom(cls, dcm: Dicom) -> "PixelSpacing":
        return cls.from_tags({tag: value for tag, value in get_tag_values(cls.get_required_tags(), dcm).items()})

    @staticmethod
    def get_required_tags() -> List[Tag]:
        return [Tag.ImagerPixelSpacing, Tag.PixelSpacing]


__all__ = [
    "Dicom",
    "ImageType",
    "PhotometricInterpretation",
    "EnumMixin",
    "Laterality",
    "ViewPosition",
    "MammogramView",
    "PixelSpacing",
]

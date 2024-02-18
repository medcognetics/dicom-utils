#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib
import inspect
import json
import logging
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field, fields, replace
from functools import cached_property, partial
from io import BytesIO, IOBase
from itertools import chain
from os import PathLike
from pathlib import Path, PosixPath
from statistics import mode
from typing import (
    Any,
    Dict,
    Final,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import pydicom
from pydicom import DataElement, Dataset, Sequence
from pydicom.multival import MultiValue
from pydicom.uid import SecondaryCaptureImageStorage


logger = logging.getLogger(__name__)


from registry import Registry

from ..dicom import Dicom
from ..tags import Tag

# Type checking fails when dataclass attr name matches a type alias.
# Import types under a different alias
from ..types import DicomKeyError, DicomValueError
from ..types import ImageType as IT
from ..types import Laterality, MammogramType, MammogramView
from ..types import PhotometricInterpretation as PI
from ..types import PixelSpacing as PixelSpacingClass
from ..types import ViewPosition, get_value, iterate_view_modifier_codes
from .helpers import SOPUID, ImageUID, SeriesUID, StudyUID
from .helpers import TransferSyntaxUID as TSUID
from .protocols import (
    SupportsDataSetInfo,
    SupportsGenerated,
    SupportsManufacturer,
    SupportsPatientAge,
    SupportsPatientID,
    SupportsSite,
    SupportsStudyDate,
    SupportsStudyUID,
    SupportsUID,
)


STANDARD_MAMMO_VIEWS: Final[Set[MammogramView]] = {
    MammogramView(Laterality.LEFT, ViewPosition.MLO),
    MammogramView(Laterality.RIGHT, ViewPosition.MLO),
    MammogramView(Laterality.LEFT, ViewPosition.CC),
    MammogramView(Laterality.RIGHT, ViewPosition.CC),
}


R = TypeVar("R", bound="FileRecord")
T = TypeVar("T")

RECORD_REGISTRY = Registry("records")
HELPER_REGISTRY = Registry("helpers")
SEP: Final[str] = "_"
DEFAULT_FILE_ID: Final = 1
FILE_ID_IDX: Final = -1


class StandardizedFilename(PosixPath):
    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls, *args, **kwargs)
        if SEP not in self.name:
            parts = self.tokenize() + [str(DEFAULT_FILE_ID)]
            self = StandardizedFilename(SEP.join(parts)).with_suffix(self.suffix)
        return self

    @property
    def prefix(self) -> str:
        return SEP.join(self.prefix_parts)

    @property
    def prefix_parts(self) -> List[str]:
        return self.tokenize()[:FILE_ID_IDX]

    @property
    def file_id(self) -> str:
        return self.tokenize()[FILE_ID_IDX]

    def with_file_id(self, val: Any) -> "StandardizedFilename":
        parts = self.tokenize()[:FILE_ID_IDX]
        parts.append(str(val))
        return StandardizedFilename(SEP.join(parts)).with_suffix(self.suffix)

    def with_prefix(self, *val: str) -> "StandardizedFilename":
        parts = list(val) + [self.file_id]
        return StandardizedFilename(SEP.join(parts)).with_file_id(self.file_id).with_suffix(self.suffix)

    def add_modifier(self, val: Any) -> "StandardizedFilename":
        parts = self.tokenize()
        parts.insert(-1, str(val))
        return StandardizedFilename(SEP.join(parts)).with_suffix(self.suffix)

    def tokenize(self) -> List[str]:
        return str(self.with_suffix("")).split(SEP)


@RECORD_REGISTRY(name="file")
@dataclass(frozen=True, order=True)
class FileRecord:
    path: Path = field(compare=True)

    def __post_init__(self):
        object.__setattr__(self, "path", Path(self.path))

    def __repr__(self) -> str:
        contents = [f"{name}={value}" for name, value in self.present_fields()]
        return f"{self.__class__.__name__}({', '.join(contents)})"

    def __hash__(self) -> int:
        return hash(self.path)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) and self.path == other.path

    def replace(self: R, **kwargs) -> R:
        return replace(self, **kwargs)

    @property
    def is_symlink(self) -> bool:
        return self.path.is_symlink()

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    @property
    def file_size(self) -> int:
        return self.path.stat().st_size

    @property
    def is_compressed(self) -> bool:
        return False

    @property
    def has_uid(self) -> bool:
        return bool(self.path)

    def get_uid(self) -> Hashable:
        return self.path.stem

    @classmethod
    def from_file(cls: Type[R], path: PathLike, helpers: Iterable["RecordHelper"] = []) -> R:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        result = cls(path)
        result = apply_helpers(path, result, helpers)
        return result

    def relative_to(self: R, target: Union[PathLike, "FileRecord"]) -> R:
        path = target.path if isinstance(target, FileRecord) else Path(target)
        # `path` is a parent of `self.path`
        if self.path.is_relative_to(path):
            return replace(self, path=self.path.relative_to(path))

        # `self.path` shares a parent with `path`
        path = path.absolute()
        self_path = self.path.absolute()
        paths_to_check = [path, *path.parents]
        for i, parent in enumerate(paths_to_check):
            if self_path.is_relative_to(parent):
                relpath = self_path.relative_to(parent)
                self_path = Path(*([".."] * i), relpath)
                break
        else:
            raise ValueError(f"Record path {self.path} is not relative to {path}")
        return replace(self, path=self_path)

    def shares_directory_with(self, other: "FileRecord") -> bool:
        return self.path.absolute().parent == other.path.absolute().parent

    def absolute(self: R) -> R:
        return replace(self, path=self.path.absolute())

    def present_fields(self) -> Iterator[Tuple[str, Any]]:
        for f in fields(self):
            value = getattr(self, f.name)
            if value != f.default:
                yield f.name, value

    def standardized_filename(self, file_id: Any = None) -> StandardizedFilename:
        file_id = str(file_id) if file_id is not None else str(self.get_uid() if self.has_uid else "1")
        file_id = file_id.replace(".", "-")
        return StandardizedFilename(self.path.name).with_file_id(file_id)

    @classmethod
    def read(
        cls,
        target: Union[PathLike, "FileRecord"],
        *args,
        helpers: Iterable["RecordHelper"] = [],
        **kwargs,
    ) -> IOBase:
        r"""Reads a DICOM file with optimized defaults for :class:`DicomFileRecord` creation.

        Args:
            path: Path to DICOM file to read

        Keyword Args:
            Overrides forwarded to :func:`pydicom.dcmread`
        """
        path = Path(target.path if isinstance(target, FileRecord) else target)
        if not path.is_file():
            raise FileNotFoundError(path)
        result = open(path, *args, **kwargs)
        result = apply_read_helpers(result, cls, helpers)
        return result

    def to_symlink(self: R, symlink_path: PathLike, overwrite: bool = False) -> R:
        r"""Create a symbolic link to the file referenced by this :class:`FileRecord`.
        The symbolic link will be relative to the location of the file referenced by this
        :class:`FileRecord`.

        Args:
            symlink_path:
                Filepath for the output symlink

            overwrite:
                If ``True``, and ``symlink_path`` is an existing symbolic link, overwrite it

        Returns:
            A new :class:`FileRecord` with ``path`` set to ``symlink_path``.
        """
        symlink_record = self.replace(path=symlink_path)
        symlink_record.path.parent.mkdir(exist_ok=True, parents=True)
        symlink_contents = Path(*self.relative_to(symlink_record).path.parts[1:])
        if overwrite:
            symlink_record.path.unlink(missing_ok=True)
        symlink_record.path.symlink_to(symlink_contents)

        resolved_path = symlink_record.path.resolve().absolute()
        real_path = self.path.resolve().absolute()
        assert resolved_path == real_path, f"{resolved_path} did not match {real_path}"
        return symlink_record

    def to_dict(self, file_id: Any = None) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        result["record_type"] = self.__class__.__name__
        module = inspect.getmodule(self.__class__)
        module = module.__name__ if module is not None else ""
        if module == "__main__":
            import warnings

            warnings.warn("TODO")
        result["record_type_import"] = module
        result["path"] = str(self.path)
        result["resolved_path"] = str(self.path.resolve())
        return result

    @staticmethod
    def from_dict(target: Union[Path, Dict[str, Any]]) -> "FileRecord":
        r"""Create a :class:`FileRecord` from a dictionary. The dictionary should have been
        created with :meth:`FileRecord.to_dict`. Keys ``record_type`` and ``record_type_import``
        are used to determine the class to instantiate.
        """
        if isinstance(target, Path):
            with open(target) as f:
                target = json.load(f)
        elif not isinstance(target, dict):
            raise TypeError(f"`target` should be a Path or dict, got {type(target)}")

        assert isinstance(target, dict)
        import_path = target["record_type_import"]
        class_name = target["record_type"]
        try:
            # get the class to be instantiated
            mod = importlib.import_module(import_path)
            cls: Type[FileRecord] = getattr(mod, class_name)
            # run classmethod for instantiation
            return cls._from_dict(target)

        except Exception as ex:
            logger.warn("Failed to create FileRecord from dict")
            logger.warn("Exception info", exc_info=ex)
            record_path = Path(target["path"])
            return FileRecord(record_path)

    @classmethod
    def _from_dict(cls: Type[R], target: Dict[str, Any]) -> R:
        # find dataclass attributes that exist in the dict
        kwargs: Dict[str, Any] = {}
        dest_fields = set(f.name for f in fields(cls))
        for k, v in target.items():
            if k in dest_fields:
                kwargs[k] = v
        record_path = Path(kwargs.pop("path"))
        rec = cls(record_path, **kwargs)
        return rec


# NOTE: record contents should follow this naming scheme:
#   * When a DICOM tag is read directly, attribute name should match tag name.
#     E.g. Tag StudyInstanceUID -> Attribute StudyInstanceUID
#   * When a one or more DICOM tags are read with additional parsing logic, attribute
#     name should differ from tag name. E.g. attribute `view_position` has advanced parsing
#     logic that reads over multiple tags
# TODO: find a way to functools.partial this dataclass decorator that works with type checker


@RECORD_REGISTRY(name="dicom", suffixes=[".dcm"])
@dataclass(frozen=True, order=False, eq=False)
class DicomFileRecord(
    FileRecord,
    SupportsStudyUID,
    SupportsStudyDate,
    SupportsManufacturer,
    SupportsUID,
    SupportsPatientID,
    SupportsPatientAge,
    SupportsGenerated,
    SupportsSite,
    SupportsDataSetInfo,
):
    r"""Data structure for storing critical information about a DICOM file.
    File IO operations on DICOMs can be expensive, so this class collects all
    required information in a single pass to avoid repeated file opening.
    """

    StudyInstanceUID: Optional[StudyUID] = None
    SeriesInstanceUID: Optional[SeriesUID] = None
    SOPInstanceUID: Optional[SOPUID] = None
    SOPClassUID: Optional[SOPUID] = None
    TransferSyntaxUID: Optional[TSUID] = None
    Modality: Optional[str] = None
    BodyPartExamined: Optional[str] = None
    PatientOrientation: Optional[List[str]] = None
    StudyDate: Optional[str] = None
    ContentDate: Optional[str] = None
    AcquisitionDate: Optional[str] = None
    SeriesDescription: Optional[str] = None
    StudyDescription: Optional[str] = None
    PatientName: Optional[str] = None
    PatientID: Optional[str] = None
    StudyID: Optional[str] = None
    Manufacturer: Optional[str] = None
    ManufacturerModelName: Optional[str] = None
    ManufacturerModelNumber: Optional[str] = None
    PresentationIntentType: Optional[str] = None
    PatientAge: Optional[str] = None
    PatientBirthDate: Optional[str] = None
    BodyPartThickness: Optional[str] = None
    InstitutionAddress: Optional[str] = None
    InstitutionName: Optional[str] = None
    TreatmentSite: Optional[str] = None
    DataSetName: Optional[str] = None
    DataSetSource: Optional[str] = None
    DataSetDescription: Optional[str] = None
    DataSetType: Optional[str] = None
    DataSetSubtype: Optional[str] = None
    PerformedProcedureStepDescription: Optional[str] = None
    AccessionNumber: Optional[str] = None

    generated: bool = False
    is_cad: bool = False

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return False

        if self.SOPInstanceUID and other.SOPInstanceUID:
            return self.same_uid_as(other)
        return self.path == other.path

    def __lt__(self, other: FileRecord) -> bool:
        if not isinstance(other, DicomFileRecord):
            return FileRecord(self.path) < other
        return str(self.SOPInstanceUID) < str(other.SOPInstanceUID)

    def __gt__(self, other: FileRecord) -> bool:
        if not isinstance(other, DicomFileRecord):
            return FileRecord(self.path) > other
        return str(self.SOPInstanceUID) > str(other.SOPInstanceUID)

    def __le__(self, other: FileRecord) -> bool:
        if not isinstance(other, DicomFileRecord):
            return FileRecord(self.path) <= other
        return str(self.SOPInstanceUID) <= str(other.SOPInstanceUID)

    def __ge__(self, other: FileRecord) -> bool:
        if not isinstance(other, DicomFileRecord):
            return FileRecord(self.path) >= other
        return str(self.SOPInstanceUID) >= str(other.SOPInstanceUID)

    def __hash__(self) -> int:
        if self.has_uid:
            return hash(self.get_uid(prefer_sop=True))
        return hash(self.path)

    def __iter__(self) -> Iterator[Tuple[Tag, Any]]:
        for f in fields(self):
            name = f.name
            value = getattr(self, name)
            if (tag := getattr(Tag, f.name, None)) is not None and value is not None:
                yield tag, value

    @cached_property
    def is_for_processing(self) -> bool:
        return (self.PresentationIntentType or "").lower() == "for processing"

    @property
    def is_compressed(self) -> bool:
        return False

    @property
    def is_secondary_capture(self) -> bool:
        return (self.SOPClassUID or "") == SecondaryCaptureImageStorage

    @cached_property
    def is_diagnostic(self) -> bool:
        if "diag" in (self.StudyDescription or "").lower():
            return True
        return False

    @cached_property
    def is_screening(self) -> bool:
        if "screening" in (self.StudyDescription or "").lower():
            return True
        return False

    @property
    def is_pr_file(self) -> bool:
        return self.Modality == "PR"

    @property
    def year(self) -> Optional[int]:
        r"""Extracts a year from ``Tag.StudyDate``.

        Returns:
            First 4 digits of ``Tag.StudyDate`` as an int, or None if a year could not be parsed
        """
        if self.StudyDate and len(self.StudyDate) > 4:
            try:
                return int(self.StudyDate[:4])
            except Exception:
                pass
        return None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        for tag, value in self:
            if not isinstance(value, Sequence):
                result[tag.name] = value
        result["secondary_capture"] = self.is_secondary_capture
        result["diagnostic"] = self.is_diagnostic
        result["screening"] = self.is_screening
        result["is_cad"] = self.is_cad
        result["generated"] = self.generated
        return result

    @classmethod
    def from_file(
        cls,
        path: PathLike,
        helpers: Iterable["RecordHelper"] = [],
        overrides: Dict[str, Any] = {},
        **kwargs,
    ) -> "DicomFileRecord":
        r"""Creates a :class:`DicomFileRecord` from a DICOM file.

        Args:
            path: Path to DICOM file
            helpers: List of :class:`RecordHelper` objects to use for parsing
            overrides: Dictionary of overrides to apply to the record

        Keyword Args:
            Forwarded to :func:`DicomFileRecord.read`
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        with cls.read(path, helpers=helpers, **kwargs) as dcm:
            result = cls.from_dicom(path, dcm, **overrides)
        result = apply_helpers(path, result, helpers)
        return result

    @classmethod
    def from_dicom(
        cls,
        path: PathLike,
        dcm: Dicom,
        helpers: Iterable["RecordHelper"] = [],
        **overrides,
    ) -> "DicomFileRecord":
        r"""Creates a :class:`DicomFileRecord` from a DICOM file.

        Args:
            path: Path to DICOM file (needed to set ``path`` attribute)
            dcm: Dicom file object
            helpers: List of :class:`RecordHelper` objects to use for parsing

        Keyword Args:
            Overrides to apply to the record
        """
        path = Path(path)
        values = {tag.name: get_value(dcm, tag, None, try_file_meta=True) for tag in cls.get_required_tags()}
        values.update(overrides)

        # pop any values that arent part of the DicomFileRecord constructor, such as intermediate tags
        keyword_values = set(f.name for f in fields(cls))
        values = {k: v for k, v in values.items() if k in keyword_values}

        rec = cls(
            path.absolute(),
            **values,
        )
        rec = apply_helpers(path.absolute(), rec, helpers)
        return rec

    @property
    def has_uid(self) -> bool:
        r"""Tests if the record has a SeriesInstanceUID or SOPInstanceUID"""
        return bool(self.SeriesInstanceUID or self.SOPInstanceUID)

    def get_uid(self, prefer_sop: bool = True) -> ImageUID:
        r"""Gets an image level UID. The UID will be chosen from SeriesInstanceUID and SOPInstanceUID,
        with preference as specified in ``prefer_sop``.
        """
        if not self.has_uid:
            raise AttributeError("DicomFileRecord has no UID")
        if prefer_sop:
            result = self.SOPInstanceUID or self.SeriesInstanceUID
        else:
            result = self.SeriesInstanceUID or self.SOPInstanceUID
        assert result is not None
        return result

    @classmethod
    def read(cls, path: PathLike, *args, helpers: Iterable["RecordHelper"] = [], **kwargs) -> Dicom:
        r"""Reads a DICOM file with optimized defaults for :class:`DicomFileRecord` creation.

        Args:
            path: Path to DICOM file to read

        Keyword Args:
            Overrides forwarded to :func:`pydicom.dcmread`
        """
        kwargs.setdefault("stop_before_pixels", True)
        kwargs.setdefault("specific_tags", cls.get_required_tags())
        stream = cast(BytesIO, FileRecord.read(path, "rb"))
        result = pydicom.dcmread(stream, *args, **kwargs)
        result = apply_read_helpers(result, cls, helpers)
        return result

    def standardized_filename(self, file_id: Optional[str] = None) -> StandardizedFilename:
        path = super().standardized_filename(file_id)
        path = path.with_prefix(self.Modality.lower() if self.Modality else "unknown")
        if self.is_secondary_capture:
            path = path.add_modifier("secondary")
        if self.is_for_processing:
            path = path.add_modifier("proc")
        if self.is_cad:
            path = path.add_modifier("cad")
        return path

    @classmethod
    def get_required_tags(cls) -> Set[Tag]:
        return {getattr(Tag, field.name) for field in fields(cls) if hasattr(Tag, field.name)}


@RECORD_REGISTRY(name="dicom-image", suffixes=[".dcm"])
@dataclass(frozen=True, order=False, eq=False)
class DicomImageFileRecord(DicomFileRecord):
    r"""Data structure for storing critical information about a DICOM file.
    File IO operations on DICOMs can be expensive, so this class collects all
    required information in a single pass to avoid repeated file opening.
    """

    TransferSyntaxUID: Optional[TSUID] = None

    Rows: Optional[int] = None
    Columns: Optional[int] = None
    NumberOfFrames: Optional[int] = None
    PhotometricInterpretation: Optional[PI] = None
    ImageType: Optional[IT] = None
    BitsStored: Optional[int] = None
    PixelSpacing: Optional[Union[str, MultiValue]] = None
    ImagerPixelSpacing: Optional[Union[str, MultiValue]] = None
    ViewCodeSequence: Optional[Dataset] = None
    ViewModifierCodeSequence: Optional[Dataset] = None
    ViewPosition: Optional[str] = None

    @property
    def is_valid_image(self) -> bool:
        return bool(self.Rows and self.Columns and self.PhotometricInterpretation)

    @property
    def is_compressed(self) -> bool:
        return bool(self.TransferSyntaxUID) and self.TransferSyntaxUID.is_compressed

    @property
    def is_volume(self) -> bool:
        return self.is_valid_image and ((self.NumberOfFrames or 1) > 1)

    @cached_property
    def is_specimen(self) -> bool:
        return "specimen" in (self.StudyDescription or "").lower()

    @property
    def image_area(self) -> Optional[int]:
        if self.Rows is not None and self.Columns is not None:
            return self.Rows * self.Columns
        return None

    @cached_property
    def is_magnified(self) -> bool:
        keywords = {"magnification", "magnified"}
        for modifier in self.view_modifier_codes:
            meaning = get_value(modifier, Tag.CodeMeaning, "").strip().lower()
            if meaning in keywords:
                return True
        return False

    @property
    def pixel_spacing(self) -> Optional[PixelSpacingClass]:
        if not self.PixelSpacing and not self.ImagerPixelSpacing:
            return None
        return PixelSpacingClass.from_tags(
            {
                Tag.PixelSpacing: self.PixelSpacing,
                Tag.ImagerPixelSpacing: self.ImagerPixelSpacing,
            }
        )

    @property
    def view_codes(self) -> Iterator[Dataset]:
        r"""Returns an iterator over all view codes"""
        if self.ViewCodeSequence is not None:
            yield from iterate_view_modifier_codes(self.ViewCodeSequence)

    @property
    def view_code_meanings(self) -> List[str]:
        r"""Returns a list of view code meanings"""
        return [get_value(modifier, Tag.CodeMeaning, "").strip().lower() for modifier in self.view_codes]

    @property
    def view_modifier_codes(self) -> Iterator[Dataset]:
        r"""Returns an iterator over all view modifier codes"""
        if self.ViewModifierCodeSequence is not None:
            yield from iterate_view_modifier_codes(self.ViewModifierCodeSequence)

    @property
    def view_modifier_code_meanings(self) -> List[str]:
        r"""Returns a list of view modifier code meanings"""
        return [get_value(modifier, Tag.CodeMeaning, "").strip().lower() for modifier in self.view_modifier_codes]

    @cached_property
    def all_view_code_meanings(self) -> Set[str]:
        r"""Returns a set of code meanings for all view or view modifier codes"""
        return {meaning for meaning in chain(self.view_code_meanings, self.view_modifier_code_meanings)}

    @classmethod
    def from_dicom(
        cls: Type[R],
        path: PathLike,
        dcm: Dicom,
        helpers: Iterable["RecordHelper"] = [],
        **overrides,
    ) -> R:
        r"""Creates a :class:`MammogramFileRecord` from a DICOM file.

        Args:
            path: Path to DICOM file (needed to set ``path`` attribute)
            dcm: Dicom file object
            modality: Optional modality override
        """
        # raise an exception if Rows/Columns aren't present
        for tag in (Tag.Rows, Tag.Columns):
            value = get_value(dcm, tag, None)
            if value is None:
                raise DicomKeyError(tag)
            elif not value:
                raise DicomValueError(tag)
        rec = cast(R, super().from_dicom(path, dcm, **overrides))
        rec = apply_helpers(rec.path, rec, helpers)
        return rec

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["magnified"] = self.is_magnified
        if self.ViewCodeSequence is not None:
            result["ViewCodeSequence"] = self.view_code_meanings
        if self.ViewModifierCodeSequence is not None:
            result["ViewCodeModifierSequence"] = self.view_modifier_code_meanings
        return result

    @classmethod
    def _from_dict(cls: Type[R], target: Dict[str, Any]) -> R:
        result = super()._from_dict(target)

        # convert view code meaning strings in dict to Sequence
        view_codes = target.get("ViewCodeSequence", [])
        if view_codes:
            result = replace(result, ViewCodeSequence=DicomImageFileRecord.make_view_code_sequence(view_codes))
        view_modifier_codes = target.get("ViewCodeModifierSequence", [])
        if view_modifier_codes:
            result = replace(
                result, ViewModifierCodeSequence=DicomImageFileRecord.make_view_code_sequence(view_modifier_codes)
            )

        return cast(R, result)

    @staticmethod
    def make_view_code(meaning: str) -> Dataset:
        vc = Dataset()
        vc[Tag.CodeMeaning] = DataElement(Tag.CodeMeaning, "ST", meaning)
        return vc

    @staticmethod
    def make_view_code_sequence(meanings: Iterable[str]) -> Sequence:
        return Sequence([DicomImageFileRecord.make_view_code(meaning) for meaning in meanings])

    def standardized_filename(self, file_id: Optional[str] = None) -> StandardizedFilename:
        path = super().standardized_filename(file_id)
        if self.is_specimen:
            path = path.add_modifier("specimen")
        return path


@RECORD_REGISTRY(name="mammogram", suffixes=[".dcm"])
@dataclass(frozen=True, order=False, eq=False)
class MammogramFileRecord(DicomImageFileRecord):
    r"""Data structure for storing critical information about a DICOM file.
    File IO operations on DICOMs can be expensive, so this class collects all
    required information in a single pass to avoid repeated file opening.
    """

    mammogram_type: MammogramType = MammogramType.UNKNOWN
    view_position: ViewPosition = ViewPosition.UNKNOWN
    laterality: Laterality = Laterality.UNKNOWN
    PaddleDescription: Optional[str] = None
    BreastImplantPresent: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.mammogram_type, str):
            object.__setattr__(self, "mammogram_type", MammogramType.from_str(self.mammogram_type))
        if isinstance(self.view_position, str):
            object.__setattr__(self, "view_position", ViewPosition.from_str(self.view_position))
        if isinstance(self.laterality, str):
            object.__setattr__(self, "laterality", Laterality.from_str(self.laterality))

    def __lt__(self, other: FileRecord) -> bool:
        return self.is_preferred_to(other)

    def __gt__(self, other: FileRecord) -> bool:
        if isinstance(other, MammogramFileRecord):
            return other.is_preferred_to(self)
        return super().__gt__(other)

    def __le__(self, other: FileRecord) -> bool:
        if isinstance(other, MammogramFileRecord):
            prefer_self = self.is_preferred_to(other)
            prefer_other = other.is_preferred_to(self)
            return prefer_self or not (prefer_self or prefer_other)
        return super().__le__(other)

    def __ge__(self, other: FileRecord) -> bool:
        if isinstance(other, MammogramFileRecord):
            prefer_self = self.is_preferred_to(other)
            prefer_other = other.is_preferred_to(self)
            return prefer_other or not (prefer_self or prefer_other)
        return super().__ge__(other)

    def is_preferred_to(self, other: FileRecord) -> bool:
        r"""Checks if this record is preferred to another record.

        Preference is determined as follows:
            * If the other record is not a mammogram, fall back to comparison by SOPInstanceUID
            * If this record is a standard view and the other record is not, this record is preferred
            * If this record is an implant displaced view and the other record is not, this record is preferred
            * If this record's image type is preferred over the other record's image type, this record is preferred
            * If this record has a higher resolution than the other record, this record is preferred
        """
        if isinstance(other, MammogramFileRecord):
            # Standard views take priority over nonstandard views
            if self.is_standard_mammo_view and not other.is_standard_mammo_view:
                return True
            # Implant displaced views take priority over implant views (only for same study)
            elif (
                self.StudyInstanceUID == other.StudyInstanceUID
                and self.is_implant_displaced
                and not other.is_implant_displaced
            ):
                return True
            # If mammograms have different types, order by type priority
            elif (
                self.mammogram_type is not None
                and other.mammogram_type is not None
                and self.mammogram_type != other.mammogram_type
            ):
                return self.mammogram_type.is_preferred_to(other.mammogram_type)
            # If mammograms have different resolutions, order by resolution
            elif self.image_area != other.image_area:
                # Higher resolution is preferred, so flip the comparison sign
                return (self.image_area or float("inf")) > (other.image_area or float("inf"))
        # Super compares by SOPInstanceUID
        return super().__lt__(other)

    @property
    def mammogram_view(self) -> MammogramView:
        return MammogramView.create(self.laterality, self.view_position)

    @property
    def is_2d(self) -> bool:
        return self.mammogram_type is not None and self.mammogram_type != MammogramType.TOMO

    @cached_property
    def is_spot_compression(self) -> bool:
        if any(s in (self.PaddleDescription or "") for s in ("SPOT", "SPT")):
            return True
        if "spot" in (self.ViewPosition or "").lower():
            return True
        return "spot compression" in self.all_view_code_meanings

    @cached_property
    def is_magnified(self) -> bool:
        return "MAG" in (self.PaddleDescription or "") or super().is_magnified

    @cached_property
    def is_implant_displaced(self) -> bool:
        return "implant displaced" in self.all_view_code_meanings

    @cached_property
    def is_tangential(self) -> bool:
        if "tan" in (self.ViewPosition or "").lower():
            return True
        return "tangential" in self.all_view_code_meanings

    @cached_property
    def is_nipple_in_profile(self) -> bool:
        if "np" in (self.ViewPosition or "").lower():
            return True
        return "nipple in profile" in self.all_view_code_meanings

    @cached_property
    def is_infra_mammary_fold(self) -> bool:
        if "imf" in (self.ViewPosition or "").lower():
            return True
        return "infra-mammary fold" in self.all_view_code_meanings

    @cached_property
    def is_anterior_compression(self) -> bool:
        if "ac" in (self.ViewPosition or "").lower():
            return True
        return "anterior compression" in self.all_view_code_meanings

    @cached_property
    def is_stereo(self) -> bool:
        r"""Check if this is a stereotactic biopsy mammogram"""
        if self.ImageType and "STEREO" in self.ImageType:
            return True

        # NOTE: this info is also in the PerformedProtocolCodeSequence.
        # However, checking the 3 places below should be sufficient
        return any(
            "stereo" in (getattr(self, tag.name) or "").lower()
            for tag in (
                Tag.StudyDescription,
                Tag.SeriesDescription,
                Tag.PerformedProcedureStepDescription,
            )
        )

    @cached_property
    def is_standard_mammo_view(self) -> bool:
        r"""Checks if this record corresponds to a standard mammography view.
        Standard mammography views are the MLO and CC views.
        """
        return (
            (self.view_position in {ViewPosition.MLO, ViewPosition.CC})
            and not self.is_spot_compression
            and not self.is_magnified
            and not self.is_secondary_capture
            and not self.is_for_processing
            and not self.is_cad
            and not self.is_stereo
            and not self.is_anterior_compression
            and not self.is_nipple_in_profile
            and not self.is_infra_mammary_fold
            and not self.is_tangential
            and not self.is_specimen
        )

    @property
    def has_implant(self) -> bool:
        return self.BreastImplantPresent == "YES"

    @classmethod
    def get_standard_mammo_view_lookup(
        cls, records: Iterable["MammogramFileRecord"]
    ) -> Dict[MammogramView, List["MammogramFileRecord"]]:
        needed_views: Dict[MammogramView, List[MammogramFileRecord]] = defaultdict(list)
        for rec in records:
            # only consider standard views
            if not isinstance(rec, MammogramFileRecord) or not rec.is_standard_mammo_view:
                continue
            key = rec.mammogram_view
            if key in STANDARD_MAMMO_VIEWS:
                needed_views[key].append(rec)
        return needed_views

    @classmethod
    def is_complete_mammo_case(cls, records: Iterable["MammogramFileRecord"]) -> bool:
        view_lookup = cls.get_standard_mammo_view_lookup(records)
        return set(view_lookup.keys()) == STANDARD_MAMMO_VIEWS

    @classmethod
    def collection_laterality(cls, records: Iterable["MammogramFileRecord"]) -> Laterality:
        left = right = False
        for rec in records:
            if not isinstance(rec, MammogramFileRecord) or rec.is_secondary_capture:
                continue
            if rec.laterality == Laterality.LEFT:
                left = True
            elif rec.laterality == Laterality.RIGHT:
                right = True

        if left and right:
            return Laterality.BILATERAL
        elif left:
            return Laterality.LEFT
        elif right:
            return Laterality.RIGHT
        else:
            return Laterality.NONE

    @classmethod
    def get_preferred_views(
        cls, col: Iterable["MammogramFileRecord"]
    ) -> Dict[MammogramView, Optional["MammogramFileRecord"]]:
        r"""Selects preferred inference views from a collection of :class:`MammogramFileRecord`s.

        Args:
            col: Collection to select from

        Returns:
            Dictionary giving the selected view (or `None` if a view could not be found) for each of the
            four standard view positions.
        """
        result: Dict[MammogramView, Optional["MammogramFileRecord"]] = {}
        col = list(col)

        # Try each standard view
        for mammo_view in STANDARD_MAMMO_VIEWS:
            lat, view_pos = mammo_view

            # Select only views that match what we're looking for.
            # We permit MLO-like or CC-like views
            def check_is_candidate(rec: MammogramFileRecord) -> bool:
                candidate = rec.mammogram_view
                assert view_pos.is_mlo_like or view_pos.is_cc_like
                laterality_match = candidate.laterality == lat
                view_match = candidate.is_mlo_like if view_pos.is_mlo_like else candidate.is_cc_like
                return view_match and laterality_match

            candidates = list(filter(check_is_candidate, col))

            # Select the most preferred image if one exists
            selection = min(candidates, default=None)
            result[mammo_view] = selection
        return result

    def get_opposing_laterality(self, col: Iterable["MammogramFileRecord"]) -> Optional["MammogramFileRecord"]:
        r"""Selects a complementary view of the opposing laterality from a collection of :class:`MammogramFileRecord`s.
        For example, the complementary view of a left MLO is a right MLO.

        Args:
            col: Collection to select from

        Returns:
            The selected view (or `None` if a view could not be found).
        """
        valid_laterality = self.laterality in {Laterality.LEFT, Laterality.RIGHT}
        valid_view = not (self.view_position is None or self.view_position.is_unknown)
        if not (valid_laterality and valid_view):
            return

        # Target laterality is opposite of this view
        target_laterality = cast(Laterality, self.laterality).opposite
        # Target view is same as this view
        target_view = cast(ViewPosition, self.view_position)

        candidates = self.get_preferred_views(
            {
                rec
                for rec in col
                if isinstance(rec, MammogramFileRecord)
                and rec.laterality == target_laterality
                and rec.view_position == target_view
            }
        )
        result = candidates.get(MammogramView(target_laterality, target_view), None)
        return cast(Optional[MammogramFileRecord], result)

    def standardized_filename(self, file_id: Optional[str] = None) -> StandardizedFilename:
        r"""Returns a standardized filename for the DICOM represented by this :class:`DicomFileRecord`.
        File name will be of the form ``{file_type}_{modifiers}_{view}_{file_id}.dcm``.

        Args:
            file_id:
                A unique identifier for this file that will be added as a postfix to the filename.
                If not provided the output of :func:`get_image_uid()` will be used.
        """
        if self.mammogram_type not in (None, MammogramType.UNKNOWN):
            filetype = self.mammogram_type.simple_name
        else:
            filetype = self.Modality.lower() if self.Modality else "uk"

        # modifiers
        # TODO make this a loop/lookup
        path = super().standardized_filename(file_id)
        modifiers: List[str] = path.prefix_parts[1:]
        if self.is_spot_compression:
            modifiers.append("spot")
        if self.is_magnified:
            modifiers.append("mag")
        if self.is_implant_displaced:
            modifiers.append("id")
        if self.is_tangential:
            modifiers.append("tan")
        if self.is_nipple_in_profile:
            modifiers.append("np")
        if self.is_infra_mammary_fold:
            modifiers.append("imf")
        if self.is_anterior_compression:
            modifiers.append("ac")
        if self.is_stereo:
            modifiers.append("stereo")

        view = f"{self.laterality.short_str}{self.view_position.short_str}"

        prefix = [filetype, view, *modifiers]
        prefix = [p for p in prefix if p]
        path = path.with_prefix(*prefix)
        return path.with_suffix(".dcm")

    @classmethod
    def get_required_tags(cls) -> Set[Tag]:
        return {
            *super().get_required_tags(),
            *Laterality.get_required_tags(),
            *ViewPosition.get_required_tags(),
            *MammogramType.get_required_tags(),
        }

    @classmethod
    def from_dicom(
        cls,
        path: PathLike,
        dcm: Dicom,
        is_sfm: bool = False,
        helpers: Iterable["RecordHelper"] = [],
        **overrides,
    ) -> "MammogramFileRecord":
        r"""Creates a :class:`MammogramFileRecord` from a DICOM file.

        Args:
            path: Path to DICOM file (needed to set ``path`` attribute)
            dcm: Dicom file object
            modality: Optional modality override
            is_sfm: Manual indicator if the mammogram is SFM instead of FFDM
        """
        result = super().from_dicom(path, dcm, **overrides)
        assert isinstance(result, cls)

        modality = result.Modality
        if modality is None:
            raise DicomKeyError(Tag.Modality)
        elif modality != "MG":
            raise DicomValueError(
                f"Modality {modality} is invalid for mammograms. If you are certain {path} is a mammogram, pass "
                "`Modality`='MG' to `from_dicom` or `overrides={'Modality': 'MG'}` to `from_file`."
            )

        laterality = Laterality.from_dicom(dcm)
        view_position = ViewPosition.from_dicom(dcm)
        # ignore modality here because it was checked above
        mammogram_type = MammogramType.from_dicom(dcm, is_sfm, ignore_modality=True)

        result = result.replace(
            laterality=laterality,
            view_position=view_position,
            mammogram_type=mammogram_type,
        )
        result = apply_helpers(result.path, result, helpers)
        return result

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["mammogram_type"] = self.mammogram_type.simple_name
        result["laterality"] = self.laterality.short_str
        result["view_position"] = self.view_position.short_str
        result["spot_compression"] = self.is_spot_compression
        result["implant_displaced"] = self.is_implant_displaced
        result["standard_mammo_view"] = self.is_standard_mammo_view
        return result


@dataclass
class RecordHelper(ABC):
    r"""A :class:`RecordHelper` implements logic that is run during :class:`FileRecord`
    creation and populates fields using custom logic.
    """

    def __call__(self, path: PathLike, rec: R) -> R:
        r"""Applies postprocessing logic to a :class:`FileRecord`.

        Args:
            path:
                Path of the
        """
        return rec

    def on_read(self, f: T, record_type: Type[FileRecord]) -> T:
        r"""Applies preprocessing logic to the object returned by :func:`FileRecord.read`.

        Args:
            f:
                Object returned by :func:`FileRecord.read`

            record_type:
                Type of :class:`FileRecord` performing the read

        Returns:
            Augmented ``f``
        """
        return f


def apply_helpers(path: PathLike, rec: R, helpers: Iterable[RecordHelper]) -> R:
    for h in helpers:
        if not isinstance(h, RecordHelper):
            raise TypeError(f"type {type(h)} is not a `RecordHelper`")
        logger.debug(f"Applying helper {type(h)} to {path}")
        rec = h(path, cast(R, rec))
        if not isinstance(rec, FileRecord):
            raise TypeError(f"Expected `FileRecord` from {type(h)}, got {type(rec)}")
    return cast(R, rec)


def apply_read_helpers(obj: T, record_type: Type[FileRecord], helpers: Iterable[RecordHelper]) -> T:
    for h in helpers:
        if not isinstance(h, RecordHelper):
            raise TypeError(f"type {type(h)} is not a `RecordHelper`")
        logger.debug(f"Applying read helper {type(obj)}")
        obj = h.on_read(obj, record_type)
    return obj


@HELPER_REGISTRY(name="patient-id-from-path")
@dataclass
class PatientIDFromPath(RecordHelper):
    r"""Helper that extracts a PatientID from the filepath.
    PatientID will be extracted as ``rec.path.parents[helper.level].name``.

    Args:
        Level in the filepath at which to extract PatientID
    """

    def __init__(self, level: int = 0):
        self.level = int(level)

    def __call__(self, path: PathLike, rec: R) -> R:
        if isinstance(rec, SupportsPatientID):
            name = Path(path).parents[self.level].name
            rec = cast(R, rec.replace(PatientID=name))
        return rec


@HELPER_REGISTRY(name="study-date-from-path")
@dataclass
class StudyDateFromPath(RecordHelper):
    r"""Helper that extracts StudyDate from the filepath.
    Study year will be extracted as ``int(rec.path.parents[helper.level].name)``, and
    the StudyDate field will be assigned as ``{year}0101``.

    Args:
        Level in the filepath at which to extract StudyDate
    """

    def __init__(self, level: int = 0):
        self.level = int(level)

    def __call__(self, path: PathLike, rec: R) -> R:
        if isinstance(rec, SupportsStudyDate):
            year = Path(path).parents[self.level].name
            date = f"{year}0101"
            rec = cast(R, rec.replace(StudyDate=date))
        return rec


@HELPER_REGISTRY(name="patient-orientation")
@dataclass
class ParsePatientOrientation(RecordHelper):
    def __call__(self, path: PathLike, rec: R) -> R:
        if isinstance(rec, MammogramFileRecord):
            po_laterality = Laterality.from_patient_orientation(rec.PatientOrientation or [])
            po_view_pos = ViewPosition.from_patient_orientation(rec.PatientOrientation or [])
            rec = cast(
                R,
                rec.replace(
                    laterality=rec.laterality or po_laterality,
                    view_position=rec.view_position or po_view_pos,
                ),
            )
        return rec


# register helpers with some typical values for `level`
for i in range(LEVELS_TO_REGISTER := 3):
    HELPER_REGISTRY(partial(PatientIDFromPath, level=i + 1), name=f"patient-id-from-path-{i+1}")
    HELPER_REGISTRY(partial(StudyDateFromPath, level=i + 1), name=f"study-date-from-path-{i+1}")


@dataclass
class DirectoryHelper(RecordHelper):
    def glob(self, path: PathLike, files_only: bool = True) -> Iterator[Path]:
        path = Path(path)
        assert path.is_dir()
        for p in path.glob("*"):
            if not files_only or p.is_file():
                yield p


@HELPER_REGISTRY(name="spot-compression")
@dataclass
class SpotCompressionHelper(DirectoryHelper):
    def __init__(self, size_delta: int = int(1e6)):
        self.size_delta = size_delta

    def __call__(self, path: PathLike, rec: R) -> R:
        if isinstance(rec, MammogramFileRecord):
            path = Path(path)
            file_sizes = {p: size for p, size in self.file_sizes(path.parent).items() if p.suffix == ".dcm"}
            if len(file_sizes) > 4 and (is_spot := rec.file_size < mode(file_sizes.values()) - self.size_delta):
                rec = cast(
                    R,
                    rec.replace(PaddleDescription=(rec.PaddleDescription or "") + " HELPER SPOT"),
                )
        return cast(R, rec)

    def file_sizes(self, path: PathLike) -> Dict[Path, int]:
        return {p: p.stat().st_size for p in self.glob(path)}


@HELPER_REGISTRY(name="modality")
@dataclass
class ModalityHelper(RecordHelper):
    r"""Helper to correct the modality of DICOM object. Some mammograms have a modality other than MG,
    which results in a record other than :class:`MammogramFileRecord` being used.
    """

    force: bool = False

    def on_read(self, f: T, record_type: Type[FileRecord]) -> T:
        if isinstance(f, Dicom) and (self.force or self._maybe_mammogram(f)):
            f.Modality = "MG"
        return cast(T, f)

    def _maybe_mammogram(self, dcm: Dicom) -> bool:
        if get_value(dcm, Tag.Modality, "").strip().lower() == "mg":
            return True
        body_part = get_value(dcm, Tag.BodyPartExamined, "").strip().lower()
        study_desc = get_value(dcm, Tag.StudyDescription, "").strip().lower()
        series_desc = get_value(dcm, Tag.SeriesDescription, "").strip().lower()
        return (body_part == "breast") or ("mammo" in study_desc + series_desc)

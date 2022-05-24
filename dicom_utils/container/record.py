#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, replace
from functools import cached_property, partial
from io import BytesIO, IOBase
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Dict,
    Final,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

import pydicom
from pydicom import Dataset, Sequence
from pydicom.uid import SecondaryCaptureImageStorage

from ..dicom import Dicom
from ..tags import Tag

# Type checking fails when dataclass attr name matches a type alias.
# Import types under a different alias
from ..types import DicomKeyError, DicomValueError
from ..types import ImageType as IT
from ..types import Laterality, MammogramType, MammogramView
from ..types import PhotometricInterpretation as PI
from ..types import ViewPosition, get_value, iterate_view_modifier_codes
from .helpers import SOPUID, ImageUID, SeriesUID, StudyUID
from .helpers import TransferSyntaxUID as TSUID
from .registry import Registry


STANDARD_MAMMO_VIEWS: Final[Set[MammogramView]] = {
    MammogramView(Laterality.LEFT, ViewPosition.MLO),
    MammogramView(Laterality.RIGHT, ViewPosition.MLO),
    MammogramView(Laterality.LEFT, ViewPosition.CC),
    MammogramView(Laterality.RIGHT, ViewPosition.CC),
}


R = TypeVar("R", bound="FileRecord")

RECORD_REGISTRY = Registry("records")
HELPER_REGISTRY = Registry("helpers")


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
        for helper in helpers:
            result = helper(path, result)
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

    def standardized_filename(self, file_id: Any = None) -> Path:
        file_id = str(file_id) if file_id is not None else str(self.get_uid() if self.has_uid else "")
        parts = [p for p in (self.path.stem, file_id) if p]
        path = Path(f"{'_'.join(parts)}{self.path.suffix}")
        return path

    @classmethod
    def read(cls, target: Union[PathLike, "FileRecord"], *args, **kwargs) -> IOBase:
        r"""Reads a DICOM file with optimized defaults for :class:`DicomFileRecord` creation.

        Args:
            path: Path to DICOM file to read

        Keyword Args:
            Overrides forwarded to :func:`pydicom.dcmread`
        """
        path = Path(target.path if isinstance(target, FileRecord) else target)
        if not path.is_file():
            raise FileNotFoundError(path)
        return open(path, *args, **kwargs)

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
        real_path = self.path.absolute()
        assert resolved_path == real_path, f"{resolved_path} did not match {real_path}"
        return symlink_record

    def to_dict(self, file_id: Any = None) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        result["record_type"] = self.__class__.__name__
        result["path"] = str(self.path)
        result["resolved_path"] = str(self.path.resolve())
        return result


@runtime_checkable
class SupportsStudyUID(Protocol):
    StudyInstanceUID: Optional[StudyUID]


# NOTE: record contents should follow this naming scheme:
#   * When a DICOM tag is read directly, attribute name should match tag name.
#     E.g. Tag StudyInstanceUID -> Attribute StudyInstanceUID
#   * When a one or more DICOM tags are read with additional parsing logic, attribute
#     name should differ from tag name. E.g. attribute `view_position` has advanced parsing
#     logic that reads over multiple tags
# TODO: find a way to functools.partial this dataclass decorator that works with type checker
@RECORD_REGISTRY(name="dicom", suffixes=[".dcm"])
@dataclass(frozen=True, order=False, eq=False)
class DicomFileRecord(FileRecord):
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
    SeriesDescription: Optional[str] = None
    StudyDescription: Optional[str] = None
    PatientName: Optional[str] = None
    PatientID: Optional[str] = None
    ManufacturerModelName: Optional[str] = None

    def __iter__(self) -> Iterator[Tuple[Tag, Any]]:
        for f in fields(self):
            name = f.name
            value = getattr(self, name)
            if (tag := getattr(Tag, f.name, None)) is not None and value is not None:
                yield tag, value

    def same_patient_as(self, other: "DicomFileRecord") -> bool:
        r"""Checks if this record references the same patient as record ``other``."""
        if self.PatientID:
            return self.PatientID == other.PatientID
        else:
            return bool(self.PatientName) and self.PatientName == other.PatientName

    def same_study_as(self, other: "DicomFileRecord") -> bool:
        r"""Checks if this record is part of the same study as record ``other``."""
        return bool(self.StudyInstanceUID) and self.StudyInstanceUID == other.StudyInstanceUID

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
        return result

    @classmethod
    def from_file(
        cls,
        path: PathLike,
        modality: Optional[str] = None,
        helpers: Iterable["RecordHelper"] = [],
        **kwargs,
    ) -> "DicomFileRecord":
        r"""Creates a :class:`DicomFileRecord` from a DICOM file.

        Args:
            path: Path to DICOM file
            modality: Optional modality override

        Keyword Args:
            Overrides forwarded to :func:`DicomFileRecord.read`
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        with cls.read(path, **kwargs) as dcm:
            result = cls.from_dicom(path, dcm, modality)
        for helper in helpers:
            result = helper(path, result)
        return result

    @classmethod
    def from_dicom(cls, path: PathLike, dcm: Dicom, modality: Optional[str] = None) -> "DicomFileRecord":
        r"""Creates a :class:`DicomFileRecord` from a DICOM file.

        Args:
            path: Path to DICOM file (needed to set ``path`` attribute)
            dcm: Dicom file object
            modality: Optional modality override
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        values = {tag.name: get_value(dcm, tag, None, try_file_meta=True) for tag in cls.get_required_tags()}
        if modality is not None:
            values["Modality"] = modality

        # pop any values that arent part of the DicomFileRecord constructor, such as intermediate tags
        keyword_values = set(f.name for f in fields(cls))
        values = {k: v for k, v in values.items() if k in keyword_values}

        return cls(
            path.absolute(),
            **values,
        )

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
    def read(cls, path: PathLike, *args, **kwargs) -> Dicom:
        r"""Reads a DICOM file with optimized defaults for :class:`DicomFileRecord` creation.

        Args:
            path: Path to DICOM file to read

        Keyword Args:
            Overrides forwarded to :func:`pydicom.dcmread`
        """
        kwargs.setdefault("stop_before_pixels", True)
        kwargs.setdefault("specific_tags", cls.get_required_tags())
        stream = cast(BytesIO, FileRecord.read(path, "rb"))
        return pydicom.dcmread(stream, *args, **kwargs)

    def standardized_filename(self, file_id: Optional[str] = None) -> Path:
        path = super().standardized_filename(file_id)
        parts = [
            self.Modality.lower() if self.Modality else "unknown",
            *str(path).split("_")[1:],
        ]
        if self.is_secondary_capture:
            parts.insert(1, "secondary")
        path = Path("_".join(p for p in parts if p))
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
    def is_magnified(self) -> bool:
        keywords = {"magnification", "magnified"}
        for modifier in self.view_modifier_codes:
            meaning = get_value(modifier, Tag.CodeMeaning, "").strip().lower()
            if meaning in keywords:
                return True
        return False

    @property
    def view_modifier_codes(self) -> Iterator[Dataset]:
        r"""Returns an iterator over all view modifier codes"""
        if self.ViewCodeSequence is not None:
            for modifier in iterate_view_modifier_codes(self.ViewCodeSequence):
                yield modifier
        if self.ViewModifierCodeSequence is not None:
            for modifier in iterate_view_modifier_codes(self.ViewModifierCodeSequence):
                yield modifier

    @classmethod
    def from_dicom(cls: Type[R], path: PathLike, dcm: Dicom, modality: Optional[str] = None) -> R:
        r"""Creates a :class:`MammogramFileRecord` from a DICOM file.

        Args:
            path: Path to DICOM file (needed to set ``path`` attribute)
            dcm: Dicom file object
            modality: Optional modality override
        """
        for tag in (Tag.Rows, Tag.Columns):
            value = get_value(dcm, tag, None)
            if value is None:
                raise DicomKeyError(tag)
            elif not value:
                raise DicomValueError(tag)
        return cast(R, super().from_dicom(path, dcm, modality))


@RECORD_REGISTRY(name="mammogram", suffixes=[".dcm"])
@dataclass(frozen=True, order=False, eq=False)
class MammogramFileRecord(DicomImageFileRecord):
    r"""Data structure for storing critical information about a DICOM file.
    File IO operations on DICOMs can be expensive, so this class collects all
    required information in a single pass to avoid repeated file opening.
    """
    mammogram_type: Optional[MammogramType] = None
    view_position: Optional[ViewPosition] = None
    laterality: Optional[Laterality] = None
    PaddleDescription: Optional[str] = None

    @property
    def mammogram_view(self) -> MammogramView:
        return MammogramView.create(self.laterality, self.view_position)

    @cached_property
    def is_spot_compression(self) -> bool:
        if "SPOT" in (self.PaddleDescription or ""):
            return True
        if "spot" in (self.ViewPosition or "").lower():
            return True
        for modifier in self.view_modifier_codes:
            meaning = get_value(modifier, Tag.CodeMeaning, "").strip().lower()
            if meaning == "spot compression":
                return True
        return False

    @cached_property
    def is_implant_displaced(self) -> bool:
        for modifier in self.view_modifier_codes:
            meaning = get_value(modifier, Tag.CodeMeaning, "").strip().lower()
            if meaning == "implant displaced":
                return True
        return False

    @property
    def is_standard_mammo_view(self) -> bool:
        r"""Checks if this record corresponds to a standard mammography view.
        Standard mammography views are the MLO and CC views.
        """
        return self.view_position in {ViewPosition.MLO, ViewPosition.CC}

    @classmethod
    def is_complete_mammo_case(cls, records: Iterable["MammogramFileRecord"]) -> bool:
        study_uid: Optional[StudyUID] = None
        needed_views: Dict[MammogramView, Optional[MammogramFileRecord]] = {k: None for k in STANDARD_MAMMO_VIEWS}
        for rec in records:
            # don't consider secondary captures
            if rec.is_secondary_capture:
                continue
            key = rec.mammogram_view
            if key in needed_views:
                needed_views[key] = rec
        return all(needed_views.values())

    def standardized_filename(self, file_id: Optional[str] = None) -> Path:
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
            filetype = self.Modality.lower()

        # modifiers
        modifiers: List[str] = []
        if self.is_spot_compression:
            modifiers.append("spot")
        if self.is_magnified:
            modifiers.append("mag")
        if self.is_implant_displaced:
            modifiers.append("id")

        view = f"{self.laterality.short_str}{self.view_position.short_str}"

        path = super().standardized_filename(file_id)
        parts = [filetype, *modifiers, view] + str(path).split("_")[1:]
        pattern = "_".join(p for p in parts if p)
        return Path(pattern).with_suffix(".dcm")

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
        cls, path: PathLike, dcm: Dicom, modality: Optional[str] = None, is_sfm: bool = False
    ) -> "MammogramFileRecord":
        r"""Creates a :class:`MammogramFileRecord` from a DICOM file.

        Args:
            path: Path to DICOM file (needed to set ``path`` attribute)
            dcm: Dicom file object
            modality: Optional modality override
            is_sfm: Manual indicator if the mammogram is SFM instead of FFDM
        """
        modality = modality or get_value(dcm, Tag.Modality, None)
        if modality is None:
            raise DicomKeyError(Tag.Modality)
        elif modality != "MG":
            raise DicomValueError(
                f"Modality {modality} is invalid for mammograms. If you are certain {path} is a mammogram, pass "
                "`modality`='MG' to `from_dicom`."
            )
        result = super().from_dicom(path, dcm, modality)
        assert isinstance(result, cls)

        laterality = Laterality.from_dicom(dcm)
        view_position = ViewPosition.from_dicom(dcm)
        # ignore modality here because it was checked above
        mammogram_type = MammogramType.from_dicom(dcm, is_sfm, ignore_modality=True)

        result = result.replace(
            laterality=laterality,
            view_position=view_position,
            mammogram_type=mammogram_type,
        )
        return result


class RecordHelper(ABC):
    r"""A :class:`RecordHelper` implements logic that is run during :class:`FileRecord`
    creation and populates fields using custom logic.
    """

    @abstractmethod
    def __call__(self, path: PathLike, rec: R) -> R:
        r"""Applies postprocessing logic to a :class:`FileRecord`.

        Args:
            path:
                Path of the
        """
        ...


@HELPER_REGISTRY(name="patient-id-from-path")
class PatientIDFromPath(RecordHelper):
    r"""Helper that extracts a PatientID from the filepath.
    PatientID will be extracted as ``rec.path.parents[helper.level].name``.

    Args:
        Level in the filepath at which to extract PatientID
    """

    def __init__(self, level: int = 0):
        self.level = int(level)

    def __call__(self, path: PathLike, rec: R) -> R:
        if isinstance(rec, DicomFileRecord):
            name = Path(path).parents[self.level].name
            rec = cast(R, rec.replace(PatientID=name))
        return rec


@HELPER_REGISTRY(name="study-date-from-path")
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
        if isinstance(rec, DicomFileRecord):
            year = Path(path).parents[self.level].name
            date = f"{year}0101"
            rec = cast(R, rec.replace(StudyDate=date))
        return rec


@HELPER_REGISTRY(name="patient-orientation")
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

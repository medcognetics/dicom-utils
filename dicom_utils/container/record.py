#!/usr/bin/env python
# -*- coding: utf-8 -*-


from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Final, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import pydicom
from pydicom import Dataset
from pydicom.uid import SecondaryCaptureImageStorage
from tqdm import tqdm

from ..dicom import Dicom
from ..tags import Tag

# Type checking fails when dataclass attr name matches a type alias.
# Import types under a different alias
from ..types import ImageType as IT
from ..types import Laterality, MammogramType, ModalityError
from ..types import PhotometricInterpretation as PI
from ..types import ViewPosition, get_value, view_modifier_code_iterator
from .helpers import SOPUID, ImageUID, SeriesUID, StudyUID
from .helpers import TransferSyntaxUID as TSUID


tags: Final = {
    Tag.SeriesInstanceUID,
    Tag.StudyInstanceUID,
    Tag.SOPInstanceUID,
    Tag.SOPClassUID,
    Tag.Modality,
    Tag.BodyPartExamined,
    Tag.TransferSyntaxUID,
    Tag.Rows,
    Tag.Columns,
    Tag.NumberOfFrames,
    Tag.PhotometricInterpretation,
    Tag.ImageType,
    Tag.ManufacturerModelName,
    Tag.SeriesDescription,
    Tag.PatientName,
    Tag.PatientID,
    Tag.PaddleDescription,
    Tag.StudyDate,
    Tag.ViewCodeSequence,
    Tag.ViewModifierCodeSequence,
    *Laterality.get_required_tags(),
    *ViewPosition.get_required_tags(),
    *MammogramType.get_required_tags(),
}

STANDARD_MAMMO_VIEWS: Final[Set[Tuple[Laterality, ViewPosition]]] = {
    (Laterality.LEFT, ViewPosition.MLO),
    (Laterality.RIGHT, ViewPosition.MLO),
    (Laterality.LEFT, ViewPosition.CC),
    (Laterality.RIGHT, ViewPosition.CC),
}


# NOTE: record contents should follow this naming scheme:
#   * When a DICOM tag is read directly, attribute name should match tag name.
#     E.g. Tag StudyInstanceUID -> Attribute StudyInstanceUID
#   * When a one or more DICOM tags are read with additional parsing logic, attribute
#     name should differ from tag name. E.g. attribute `view_position` has advanced parsing
#     logic that reads over multiple tags
@dataclass(frozen=True, order=True)
class FileRecord:
    r"""Data structure for storing critical information about a DICOM file.
    File IO operations on DICOMs can be expensive, so this class collects all
    required information in a single pass to avoid repeated file opening.
    """
    path: Path = field(compare=True)
    StudyInstanceUID: Optional[StudyUID]
    SeriesInstanceUID: Optional[SeriesUID]
    SOPInstanceUID: Optional[SOPUID]

    TransferSyntaxUID: Optional[TSUID]

    SOPClassUID: Optional[SOPUID] = None
    Modality: Optional[str] = None
    BodyPartExamined: Optional[str] = None
    StudyDate: Optional[str] = None

    Rows: Optional[int] = None
    Columns: Optional[int] = None
    NumberOfFrames: Optional[int] = None
    PhotometricInterpretation: Optional[PI] = None
    ImageType: Optional[IT] = None
    ViewCodeSequence: Optional[Dataset] = None
    ViewModifierCodeSequence: Optional[Dataset] = None
    mammogram_type: Optional[MammogramType] = None
    ManufacturerModelName: Optional[str] = None
    SeriesDescription: Optional[str] = None
    PaddleDescription: Optional[str] = None
    view_position: ViewPosition = ViewPosition.UNKNOWN
    laterality: Laterality = Laterality.UNKNOWN

    PatientName: Optional[str] = None
    PatientID: Optional[str] = None

    def __hash__(self) -> int:
        return hash(self.path)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, FileRecord) and self.path == other.path

    def __post_init__(self):
        if self.Modality not in (None, "MG") and self.mammogram_type is not None:
            raise ValueError("`mammogram_type` should be None when Modality != MG. " f"Found {self.mammogram_type}")

    def same_patient_as(self, other: "FileRecord") -> bool:
        r"""Checks if this record references the same patient as record ``other``."""
        if self.PatientID:
            return self.PatientID == other.PatientID
        else:
            return bool(self.PatientName) and self.PatientName == other.PatientName

    def same_study_as(self, other: "FileRecord") -> bool:
        r"""Checks if this record is part of the same study as record ``other``."""
        return bool(self.StudyInstanceUID) and self.StudyInstanceUID == other.StudyInstanceUID

    @property
    def is_image(self) -> bool:
        return bool(self.Rows and self.Columns and self.PhotometricInterpretation)

    @property
    def is_volume(self) -> bool:
        return self.is_image and ((self.NumberOfFrames or 1) > 1)

    @property
    def is_mammogram(self) -> bool:
        return self.Modality == "MG" and self.is_image and not self.is_secondary_capture

    @cached_property
    def is_spot_compression(self) -> bool:
        if not self.is_mammogram:
            return False
        if "SPOT" in (self.PaddleDescription or ""):
            return True
        for modifier in self.view_modifier_codes:
            meaning = get_value(modifier, Tag.CodeMeaning, "").strip().lower()
            if meaning == "spot compression":
                return True
        return False

    @property
    def is_secondary_capture(self) -> bool:
        return (self.SOPClassUID or "") == SecondaryCaptureImageStorage

    @property
    def is_pr_file(self) -> bool:
        return self.Modality == "PR"

    @property
    def is_ultrasound(self) -> bool:
        return self.Modality == "US" and self.is_image and not self.is_secondary_capture

    @property
    def is_tomo(self) -> bool:
        return self.is_mammogram and self.mammogram_type == MammogramType.TOMO

    @property
    def is_synthetic_view(self) -> bool:
        return self.is_mammogram and self.mammogram_type == MammogramType.SVIEW

    @property
    def is_ffdm(self) -> bool:
        return self.is_mammogram and self.mammogram_type == MammogramType.FFDM

    @property
    def is_sfm(self) -> bool:
        return self.is_mammogram and self.mammogram_type == MammogramType.SFM

    @property
    def is_standard_mammo_view(self) -> bool:
        r"""Checks if this record corresponds to a standard mammography view.
        Standard mammography views are the MLO and CC views.
        """
        return self.is_mammogram and self.view_position in {ViewPosition.MLO, ViewPosition.CC}

    @classmethod
    def is_complete_mammo_case(cls, records: Iterable["FileRecord"]) -> bool:
        study_uid: Optional[StudyUID] = None
        needed_views: Dict[Tuple[Laterality, ViewPosition], Optional[FileRecord]] = {
            k: None for k in STANDARD_MAMMO_VIEWS
        }
        for rec in records:
            key = (rec.laterality, rec.view_position)
            if key in needed_views:
                needed_views[key] = rec
        return all(needed_views.values())

    @cached_property
    def is_magnified(self) -> bool:
        keywords = {"magnification", "magnified"}
        for modifier in self.view_modifier_codes:
            meaning = get_value(modifier, Tag.CodeMeaning, "").strip().lower()
            if meaning in keywords:
                return True
        return False

    @cached_property
    def is_implant_displaced(self) -> bool:
        if not self.is_mammogram:
            return False
        for modifier in self.view_modifier_codes:
            meaning = get_value(modifier, Tag.CodeMeaning, "").strip().lower()
            if meaning == "implant displaced":
                return True
        return False

    @property
    def file_size(self) -> int:
        return self.path.stat().st_size

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

    @property
    def view_modifier_codes(self) -> Iterator[Dataset]:
        r"""Returns an iterator over all view modifier codes"""
        if self.ViewCodeSequence is not None:
            for modifier in view_modifier_code_iterator(self.ViewCodeSequence):
                yield modifier
        if self.ViewModifierCodeSequence is not None:
            for modifier in view_modifier_code_iterator(self.ViewModifierCodeSequence):
                yield modifier

    def standardized_filename(self, file_id: Optional[str] = None) -> Path:
        r"""Returns a standardized filename for the DICOM represented by this :class:`FileRecord`.
        File name will be of the form ``{file_type}_{modifiers}_{view}_{file_id}.dcm``.

        Args:
            file_id:
                A unique identifier for this file that will be added as a postfix to the filename.
                If not provided the output of :func:`get_image_uid()` will be used.
        """
        # file type
        if self.is_pr_file:
            filetype = "pr"
        elif self.is_ultrasound:
            filetype = "us"
        elif self.is_ffdm:
            filetype = "ffdm"
        elif self.is_synthetic_view:
            filetype = "synth"
        elif self.is_tomo:
            filetype = "tomo"
        else:
            # TODO we could read modality and use that as a filetype
            filetype = "unknown"

        # modifiers
        modifiers: List[str] = []
        if self.is_spot_compression:
            modifiers.append("spot")
        if self.is_magnified:
            modifiers.append("mag")
        if self.is_implant_displaced:
            modifiers.append("id")

        # view
        if self.is_mammogram:
            view = f"{self.laterality.short_str}{self.view_position.short_str}"
        else:
            view = ""

        uid = self.get_image_uid() if file_id is None else file_id
        parts = [part for part in (filetype, *modifiers, view, uid) if part]
        pattern = "_".join(parts)
        return Path(pattern).with_suffix(".dcm")

    @classmethod
    def create(cls, path: PathLike, is_sfm: bool = False) -> "FileRecord":
        r"""Creates a :class:`FileRecord` from a DICOM file.

        Args:
            path: Path to DICOM file
            is_sfm: Manual identifier if the target is a scan film mammogram.

        Keyword Args:
            Overrides forwarded to :func:`pydicom.dcmread`
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)

        with cls.read(path) as dcm:
            values = {tag.name: getattr(dcm, tag.name, None) for tag in tags}

            # overrides for tags that require additional parsing
            for key in ("Rows", "Columns", "NumberOfFrames"):
                values[key] = int(values[key]) if values[key] else None
            values["ImageType"] = IT.from_dicom(dcm)
            values["TransferSyntaxUID"] = dcm.file_meta.get("TransferSyntaxUID", None)

            # attributes that don't correspond directly to a DICOM tag
            try:
                mammogram_type = MammogramType.from_dicom(dcm, is_sfm=is_sfm)
            except ModalityError:
                mammogram_type = None
            view_position = ViewPosition.from_dicom(dcm)
            laterality = Laterality.from_dicom(dcm)

        # pop any values that arent part of the FileRecord constructor, such as intermediate tags
        values = {k: v for k, v in values.items() if k in cls.__dataclass_fields__.keys()}

        return cls(
            path.absolute(),
            mammogram_type=mammogram_type,
            view_position=view_position,
            laterality=laterality,
            **values,
        )

    @property
    def has_image_uid(self) -> bool:
        r"""Tests if the record has a SeriesInstanceUID or SOPInstanceUID"""
        return bool(self.SeriesInstanceUID or self.SOPInstanceUID)

    def get_image_uid(self, prefer_sop: bool = True) -> ImageUID:
        r"""Gets an image level UID. The UID will be chosen from SeriesInstanceUID and SOPInstanceUID,
        with preference as specified in ``prefer_sop``.
        """
        if not self.has_image_uid:
            raise AttributeError("FileRecord has no UID")
        if prefer_sop:
            result = self.SOPInstanceUID or self.SeriesInstanceUID
        else:
            result = self.SeriesInstanceUID or self.SOPInstanceUID
        assert result is not None
        return result

    @classmethod
    def read(cls, path: PathLike, **kwargs) -> Dicom:
        r"""Reads a DICOM file with optimized defaults for :class:`FileRecord` creation.

        Args:
            path: Path to DICOM file to read

        Keyword Args:
            Overrides forwarded to :func:`pydicom.dcmread`
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        kwargs.setdefault("stop_before_pixels", True)
        kwargs.setdefault("specific_tags", tags)
        return pydicom.dcmread(path, **kwargs)


def record_iterator(
    files: Sequence[PathLike], jobs: Optional[int] = None, use_bar: bool = True, threads: bool = False
) -> Iterator[FileRecord]:
    r"""Produces :class:`FileRecord` instances by iterating over an input list of files. If a
    :class:`FileRecord` cannot be created from a path, it will be ignored.

    Args:
        files:
            List of paths to iterate over

        jobs:
            Number of parallel processes to use

        use_bar:
            If ``False``, don't show a tqdm progress bar

        threads:
            If ``True``, use a :class:`ThreadPoolExecutor`. Otherwise, use a :class:`ProcessPoolExecutor`

    Returns:
        Iterator of :class:`FileRecord`s
    """
    files = list(Path(p) for p in files if Path(p).is_file())
    bar = tqdm(desc="Scanning files", total=len(files), unit="file", disable=(not use_bar))

    Pool = ThreadPoolExecutor if threads else ProcessPoolExecutor
    with Pool(jobs) as p:
        futures = [p.submit(FileRecord.create, path) for path in files]
        for f in futures:
            f.add_done_callback(lambda _: bar.update(1))
        for f in futures:
            if f.exception():
                continue
            elif record := f.result():
                yield record
    bar.close()


class RecordCollection:
    r"""Data stucture for organizing :class:`FileRecord` instances, indexed by various attriburtes."""

    def __init__(self):
        self._lookup: Dict[Path, FileRecord] = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(length={len(self)})"

    def __len__(self) -> int:
        r"""Returns the total number of contained records"""
        return len(self._lookup)

    @property
    def path_lookup(self) -> Dict[Path, FileRecord]:
        r"""Returns a dictionary mapping a filepath to a :class:`FileRecord`"""
        return self._lookup

    @cached_property
    def study_lookup(self) -> Dict[StudyUID, Set[FileRecord]]:
        r"""Returns a dictionary mapping a StudyInstanceUID to a set of :class:`FileRecord`s with
        that StudyInstanceUID. Records missing a StudyInstanceUID are ignored.
        """
        result: Dict[StudyUID, Set[FileRecord]] = {}
        for record in self._lookup.values():
            if record.StudyInstanceUID is None:
                continue
            record_set = result.get(record.StudyInstanceUID, set())
            record_set.add(record)
            result[record.StudyInstanceUID] = record_set
        return result

    @classmethod
    def from_dir(
        cls,
        path: PathLike,
        pattern: str = "*",
        jobs: Optional[int] = None,
        use_bar: bool = True,
        threads: bool = False,
        ignore_patterns: Sequence[str] = [],
    ) -> "RecordCollection":
        r"""Create a :class:`RecordCollection` from files in a directory matching a wildcard.
        If a :class:`FileRecord` cannot be created for a file, that file is silently excluded
        from the collection.

        Args:
            path:
                Path to the directory to search

            pattern:
                Glob pattern for matching files

            jobs:
                Number of parallel jobs to use

            use_bar:
                If ``False``, don't show a tqdm progress bar

            threads:
                If ``True``, use a :class:`ThreadPoolExecutor`. Otherwise, use a :class:`ProcessPoolExecutor`

            ignore_patterns:
                Strings indicating files that should be ignored. Matching is a simple ``str(pattern) in str(filepath)``.
        """
        path = Path(path)
        if not path.is_dir():
            raise NotADirectoryError(path)
        files = [p for p in path.rglob(pattern) if not any(ignore in str(p) for ignore in ignore_patterns)]
        return cls.from_files(files, jobs, use_bar, threads)

    @classmethod
    def from_files(
        cls,
        files: Sequence[PathLike],
        jobs: Optional[int] = None,
        use_bar: bool = True,
        threads: bool = False,
    ) -> "RecordCollection":
        r"""Create a :class:`RecordCollection` from a list of files.
        If a :class:`FileRecord` cannot be created for a file, that file is silently excluded
        from the collection.

        Args:
            files:
                List of files to create records from

            jobs:
                Number of parallel jobs to use

            use_bar:
                If ``False``, don't show a tqdm progress bar

            threads:
                If ``True``, use a :class:`ThreadPoolExecutor`. Otherwise, use a :class:`ProcessPoolExecutor`

            ignore_patterns:
                Strings indicating files that should be ignored. Matching is a simple ``str(pattern) in str(filepath)``.
        """
        collection = cls()
        for record in record_iterator(files, jobs, use_bar, threads):
            collection._lookup[record.path] = record
        return collection

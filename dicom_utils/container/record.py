#!/usr/bin/env python
# -*- coding: utf-8 -*-


from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Final, Iterator, List, Optional, Sequence, Set, cast

import pydicom
from pydicom import Dataset
from pydicom.uid import SecondaryCaptureImageStorage
from tqdm import tqdm
from os import PathLike

from ..tags import Tag

# Type checking fails when dataclass attr name matches a type alias.
# Import types under a different alias
from ..types import ImageType as IT
from ..types import Laterality, MammogramType, ModalityError
from ..types import PhotometricInterpretation as PI
from ..types import ViewPosition, view_code_iterator, view_modifier_code_iterator, get_value
from ..dicom import Dicom
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

    @property
    def same_patient_as(self, other: "FileRecord") -> bool:
        if self.PatientID:
            return self.PatientID == other.PatientID
        else: 
            return self.PatientName and self.PatientName == other.PatientName

    @property
    def same_study_as(self, other: "FileRecord") -> bool:
        return self.StudyInstanceUID and self.StudyInstanceUID == other.StudyInstanceUID

    @property
    def is_secondary_capture(self) -> bool:
        return (self.SOPClassUID or "") == SecondaryCaptureImageStorage

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
    def is_standard_view(self) -> bool:
        return self.is_mammogram and self.view_position in {ViewPosition.MLO, ViewPosition.CC}

    @cached_property
    def is_magnified(self) -> bool:
        for modifier in self.view_modifier_codes:
            meaning = get_value(modifier, Tag.CodeMeaning, "").strip().lower()
            if meaning == "magnification":
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
        if self.StudyDate and len(self.StudyDate) > 4:
            return self.StudyDate[:4]
        return None

    @property
    def view_modifier_codes(self) -> Iterator[Dataset]:
        if self.ViewCodeSequence is not None:
            for modifier in view_modifier_code_iterator(self.ViewCodeSequence):
                yield modifier
        if self.ViewModifierCodeSequence is not None:
            for modifier in view_modifier_code_iterator(self.ViewModifierCodeSequence):
                yield modifier

    def standardized_filename(self, file_id: Optional[str] = None) -> Path:
        uid = self.get_image_uid() if file_id is None else file_id
        if self.is_pr_file:
            prefix = "pr"
        elif self.is_ultrasound:
            prefix = "us"
        elif self.is_ffdm:
            prefix = "ffdm"
        elif self.is_synthetic_view:
            prefix = "synth"
        elif self.is_tomo:
            prefix = "tomo"
        else:
            # TODO we could read modality and use that as a prefix
            prefix = "unkown"

        if self.is_spot_compression:
            prefix += "_spot"
        if self.is_magnified:
            prefix += "_mag"
        if self.is_implant_displaced:
            prefix += "_id"

        if self.is_mammogram:
            view_info = f"{self.laterality.short_str}{self.view_position.short_str}"
            if view_info:
                prefix += f"_{view_info}"

        return Path(f"{prefix}_{uid}").with_suffix(".dcm")


    @classmethod
    def create(cls, path: Path, is_sfm: bool = False) -> "FileRecord":
        if not path.is_file():
            raise FileNotFoundError(path)

        with cls.read(path) as dcm:
            values = {tag.name: getattr(dcm, tag.name, None) for tag in tags}
            for key in ("Rows", "Columns", "NumberOfFrames"):
                values[key] = int(values[key]) if values[key] else None
            values["ImageType"] = IT.from_dicom(dcm)
            try:
                mammogram_type = MammogramType.from_dicom(dcm, is_sfm=is_sfm)
            except ModalityError:
                mammogram_type = None

            values["TransferSyntaxUID"] = dcm.file_meta.get("TransferSyntaxUID", None)
            view_position = ViewPosition.from_dicom(dcm)
            laterality = Laterality.from_dicom(dcm)

        # pop any values that arent part of the FileRecord constructor
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
        if not path.is_file():
            raise FileNotFoundError(path)
        kwargs.setdefault("stop_before_pixels", True)
        kwargs.setdefault("specific_tags", tags)
        return pydicom.dcmread(path, **kwargs)


def record_iterator(
    files: Sequence[Path], jobs: Optional[int] = None, use_bar: bool = True, threads: bool = False
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
    files = list(p for p in files if p.is_file())
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
        path: Path,
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
        if not path.is_dir():
            raise NotADirectoryError(path)
        files = [p for p in path.rglob(pattern) if not any(ignore in str(p) for ignore in ignore_patterns)]
        return cls.from_files(files, jobs, use_bar, threads)

    @classmethod
    def from_files(
        cls,
        files: Sequence[Path],
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

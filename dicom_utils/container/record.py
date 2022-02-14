#!/usr/bin/env python
# -*- coding: utf-8 -*-


from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set

import pydicom
from tqdm import tqdm

from ..tags import Tag

# Type checking fails when dataclass attr name matches a type alias.
# Import types under a different alias
from ..types import ImageType
from ..types import PhotometricInterpretation as PI
from ..types import SimpleImageType as SIT
from .helpers import SeriesUID, StudyUID
from .helpers import TransferSyntaxUID as TSUID


tags: List[Any] = [
    Tag.SeriesInstanceUID,
    Tag.StudyInstanceUID,
    Tag.SOPInstanceUID,
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
]


@dataclass(frozen=True)
class FileRecord:
    r"""Data structure for storing critical information about a DICOM file.
    File IO operations on DICOMs can be expensive, so this class collects all
    required information in a single pass to avoid repeated file opening.
    """
    path: Path
    StudyInstanceUID: Optional[StudyUID]
    SeriesInstanceUID: Optional[SeriesUID]
    SOPInstanceUID: Optional[SeriesUID]

    TransferSyntaxUID: Optional[TSUID]

    Rows: Optional[int] = None
    Columns: Optional[int] = None
    NumberOfFrames: Optional[int] = None
    PhotometricInterpretation: Optional[PI] = None
    SimpleImageType: Optional[SIT] = None
    ManufacturerModelName: Optional[str] = None
    SeriesDescription: Optional[str] = None

    PatientName: Optional[str] = None
    PatientID: Optional[str] = None

    def __hash__(self) -> int:
        return hash(self.path)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, FileRecord) and self.path == other.path

    @property
    def is_image(self) -> bool:
        return bool(self.Rows and self.Columns and self.PhotometricInterpretation)

    @property
    def is_volume(self) -> bool:
        return self.is_image and ((self.NumberOfFrames or 1) > 1)

    @property
    def file_size(self) -> int:
        return self.path.stat().st_size

    @classmethod
    def create(cls, path: Path) -> "FileRecord":
        if not path.is_file():
            raise FileNotFoundError(path)

        with pydicom.dcmread(path, stop_before_pixels=True, specific_tags=tags) as dcm:
            values = {tag.name: getattr(dcm, tag.name, None) for tag in tags}
            for key in ("Rows", "Columns", "NumberOfFrames"):
                values[key] = int(values[key]) if values[key] else None
            img_type = ImageType.from_dicom(dcm).to_simple_image_type()
            values["TransferSyntaxUID"] = dcm.file_meta.get("TransferSyntaxUID", None)

        values.pop("ImageType")
        return cls(path, SimpleImageType=img_type, **values)


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

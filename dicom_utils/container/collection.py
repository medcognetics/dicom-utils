#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    cast,
)

from tqdm import tqdm

from ..dicom import Dicom

# Type checking fails when dataclass attr name matches a type alias.
# Import types under a different alias
from .record import HELPER_REGISTRY, RECORD_REGISTRY, DicomFileRecord, FileRecord, RecordHelper, SupportsStudyUID


class RecordCreator:
    r"""RecordCreators examine candidate files and determine which :class:`FileRecord` subclass, if any,
    should be created for the candidate. A :class:`FileRecord` subclass will be chosen by trying callables
    in ``functions`` until one succeeds without raising an exception. Subclasses will be tried before
    their parents.

    Args:
        functions:
            Iterable of names for registered :class:`FileRecord`s to try when creating records.

        helpers:
            Iterable of names for registered :class:`RecordHelper`s to use when creating records.
    """

    def __init__(
        self,
        functions: Optional[Iterable[str]] = None,
        helpers: Optional[Iterable[str]] = [],
    ):
        self.functions = RECORD_REGISTRY.available_keys() if functions is None else functions
        self.helpers = [cast(Type[RecordHelper], HELPER_REGISTRY.get(h))() for h in helpers]

    def __call__(self, path: Path) -> FileRecord:
        result = FileRecord.from_file(path)
        dcm: Optional[Dicom] = None
        for dtype in self.iterate_types_to_try(path):
            try:
                # NOTE: we want to avoid repeatedly opening a DICOM file for each invalid subclass.
                # As such, open `dcm` once with all tags present and reuse it
                if issubclass(dtype, DicomFileRecord):
                    if dcm is None:
                        dcm = dtype.read(path, specific_tags=None)
                    result = dtype.from_dicom(
                        path,
                        dcm,
                    )
                else:
                    result = dtype.from_file(path)
                break
            except Exception:
                pass

        for helper in self.helpers:
            result = helper(path, result)
        return result

    def iterate_types_to_try(self, path: PathLike) -> Iterator[Type[FileRecord]]:
        r"""Iterates over the :class:`FileRecord` subclasses to try when creating ``path``
        in the order in which they should be tried.
        """
        path = Path(path)
        candidates: List[Type[FileRecord]] = []

        # exclude candidates by file extension if `path` has an extension
        for name in self.functions:
            entry = cast(Dict[str, Any], RECORD_REGISTRY.get(name, with_metadata=True))
            dtype = entry["fn"]
            assert isinstance(dtype, type)
            assert issubclass(dtype, FileRecord)
            suffixes = set(entry.get("suffixes", []))
            path_suffix = path.suffix.lower()
            valid_candidate = not path_suffix or not suffixes or path_suffix in suffixes
            if valid_candidate:
                candidates.append(dtype)

        dicom_candidates = set(self.filter_dicom_types(candidates))
        non_dicom_candidates = set(candidates) - dicom_candidates

        for c in self.sort_by_subclass_level(dicom_candidates):
            yield c
        for c in self.sort_by_subclass_level(non_dicom_candidates):
            yield c

    @classmethod
    def sort_by_subclass_level(cls, types: Iterable[Type[FileRecord]]) -> List[Type[FileRecord]]:
        return sorted(types, key=lambda t: len(t.__mro__), reverse=True)

    @classmethod
    def filter_dicom_types(cls, types: Iterable[Type[FileRecord]]) -> List[Type[DicomFileRecord]]:
        return [t for t in types if issubclass(t, DicomFileRecord)]


def record_iterator(
    files: Sequence[PathLike],
    jobs: Optional[int] = None,
    use_bar: bool = True,
    threads: bool = False,
    record_types: Optional[Iterable[str]] = None,
    helpers: Iterable[str] = [],
    ignore_exceptions: bool = False,
) -> Iterator[FileRecord]:
    r"""Produces :class:`FileRecord` instances by iterating over an input list of files. If a
    :class:`FileRecord` cannot be from_filed from a path, it will be ignored.

    Args:
        files:
            List of paths to iterate over

        jobs:
            Number of parallel processes to use

        use_bar:
            If ``False``, don't show a tqdm progress bar

        threads:
            If ``True``, use a :class:`ThreadPoolExecutor`. Otherwise, use a :class:`ProcessPoolExecutor`

        record_types:
            List of registered names for :class:`FileRecord` types to try

        helpers:
            Iterable of registered names for :class:`RecordHelper`s to use when creating records

        ignore_exceptions:
            If ``False``, any exceptions raised during record creation will not be suppressed. By default,
            exceptions are silently ignored and records will not be produced for failing files.

    Returns:
        Iterator of :class:`FileRecord`s
    """
    files = list(Path(p) for p in files if Path(p).is_file())
    creator = RecordCreator(record_types, helpers)
    bar = tqdm(desc="Scanning files", total=len(files), unit="file", disable=(not use_bar))

    Pool = ThreadPoolExecutor if threads else ProcessPoolExecutor
    with Pool(jobs) as p:
        futures = [p.submit(creator, path) for path in files]
        for f in futures:
            f.add_done_callback(lambda _: bar.update(1))
        for f in futures:
            if f.exception() and not ignore_exceptions:
                raise cast(Exception, f.exception())
            elif record := f.result():
                yield record
    bar.close()


T = TypeVar("T", bound=Hashable)
C = TypeVar("C", bound="RecordCollection")


class RecordCollection:
    r"""Data stucture for organizing :class:`FileRecord` instances, indexed by various attriburtes."""

    def __init__(self, records: Iterable[FileRecord] = set()):
        self.records: Set[FileRecord] = set(records)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(length={len(self)})"

    def __len__(self) -> int:
        r"""Returns the total number of contained records"""
        return len(self.records)

    def __iter__(self) -> Iterator[FileRecord]:
        for rec in self.records:
            yield rec

    def __contains__(self, x: Any) -> bool:
        return x in self.records

    def add(self, rec: FileRecord) -> None:
        return self.records.add(rec)

    def union(self: C, other: C) -> C:
        return self.__class__(self.records.union(other.records))

    def intersection(self: C, other: C) -> C:
        return self.__class__(self.records.intersection(other.records))

    def difference(self: C, other: C) -> C:
        return self.__class__(self.records.difference(other.records))

    def apply(self: C, func: Callable[[FileRecord], FileRecord]) -> C:
        records = {func(rec) for rec in self.records}
        return self.__class__(records)

    def filter(self: C, func: Callable[[FileRecord], bool]) -> C:
        records = {rec for rec in self.records if func(rec)}
        return self.__class__(records)

    def group_by(self: C, func: Callable[[FileRecord], T]) -> Dict[T, C]:
        result: Dict[T, C] = {}
        for record in self.records:
            key = func(record)
            result.setdefault(key, self.__class__()).add(record)
        return result

    def parent_dirs(self, offset: int = 0) -> Set[Path]:
        r"""Gets the set of unique parent directories at level ``offset``."""
        return {rec.path.parents[offset] for rec in self.records}

    def common_parent_dir(self) -> Optional[Path]:
        parents = self.parent_dirs()
        proto = next(iter(parents))
        while proto.parent != proto:
            if all(p.is_relative_to(proto) for p in parents):
                return proto
        return None

    def standardized_filenames(self) -> Iterator[Tuple[Path, FileRecord]]:
        counter: Dict[str, int] = {}
        for rec in self:
            path = rec.standardized_filename("")
            prefix = "_".join(str(path).split("_")[:-1])
            count = counter.get(prefix, 1)
            path = rec.standardized_filename(str(count))
            yield path, rec
            counter[prefix] = count + 1

    def prune_duplicates(self: C, func: Callable[[FileRecord], T]) -> C:
        groups = self.group_by(func)
        return self.__class__({next(iter(g)) for g in groups.values()})

    @property
    def sort_key(self) -> Hashable:
        r"""Gets a key that can be used to determinstically sort multiple :class:`RecordCollection`s.
        If at least one member record has a StudyInstanceUID, then the first element of the sorted
        StudyInstanceUID set will be used as a sort key. If the collection is empty, ``None`` will be used
        as a sort key. Otherwise, the first element of the sorted path set will be used as a sort key.
        """
        study_uids = sorted(
            rec.StudyInstanceUID
            for rec in self
            if isinstance(rec, SupportsStudyUID) and rec.StudyInstanceUID is not None
        )
        if study_uids:
            return next(iter(study_uids))

        paths = sorted(rec.path for rec in self)
        if paths:
            return next(iter(paths))

        return None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        result["sort_key"] = str(self.sort_key)
        result["records"] = {}
        for name, record in self.standardized_filenames():
            result["records"][str(name)] = record.to_dict()
        return result

    def to_json(self, path: PathLike, indent: int = 4) -> None:
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, sort_keys=True, indent=indent)

    @classmethod
    def from_dir(
        cls: Type[C],
        path: PathLike,
        pattern: str = "*",
        jobs: Optional[int] = None,
        use_bar: bool = True,
        threads: bool = False,
        ignore_patterns: Sequence[str] = [],
        record_types: Optional[Iterable[str]] = None,
        helpers: Iterable[str] = [],
    ) -> C:
        r"""Create a :class:`RecordCollection` from files in a directory matching a wildcard.
        If a :class:`FileRecord` cannot be from_filed for a file, that file is silently excluded
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
        return cls.from_files(files, jobs, use_bar, threads, record_types, helpers)

    @classmethod
    def from_files(
        cls: Type[C],
        files: Sequence[PathLike],
        jobs: Optional[int] = None,
        use_bar: bool = True,
        threads: bool = False,
        record_types: Optional[Iterable[str]] = None,
        helpers: Iterable[str] = [],
    ) -> C:
        r"""Create a :class:`RecordCollection` from a list of files.
        If a :class:`FileRecord` cannot be from_filed for a file, that file is silently excluded
        from the collection.

        Args:
            files:
                List of files to from_file records from

            jobs:
                Number of parallel jobs to use

            use_bar:
                If ``False``, don't show a tqdm progress bar

            threads:
                If ``True``, use a :class:`ThreadPoolExecutor`. Otherwise, use a :class:`ProcessPoolExecutor`

            ignore_patterns:
                Strings indicating files that should be ignored. Matching is a simple ``str(pattern) in str(filepath)``.
        """
        collection = cls(record_iterator(files, jobs, use_bar, threads, record_types, helpers))
        return collection

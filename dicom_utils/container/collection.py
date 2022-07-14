#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
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
    overload,
)

from registry import Registry
from tqdm import tqdm

from ..dicom import Dicom

# Type checking fails when dataclass attr name matches a type alias.
# Import types under a different alias
from .record import HELPER_REGISTRY, RECORD_REGISTRY, DicomFileRecord, FileRecord, RecordHelper


logger = logging.getLogger(__name__)

FILTER_REGISTRY = Registry("filters")


@dataclass
class RecordFilter:
    r""":class:`RecordFilter` defines a filter function that can be used to filter :class`FileRecord`s during
    the record discovery phase.

    The following filter hooks are available:
        * :func:`path_is_valid` - Checks a path before any :class:`FileRecord` creation is attempted.
        * :func:`record_is_valid` - Checks the created :class:`FileRecord`

    If either of the filter hooks return ``False``, the record will not be included in the iterator of
    discovered :class:`FileRecord`s.
    """

    def __call__(self, target: Union[Path, FileRecord]) -> bool:
        if isinstance(target, Path):
            valid = self.path_is_valid(target)
        elif isinstance(target, FileRecord):
            valid = self.record_is_valid(target)
        else:
            raise TypeError(type(target))
        return valid

    def path_is_valid(self, path: Path) -> bool:
        return True

    def record_is_valid(self, rec: FileRecord) -> bool:
        return True


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
        modality: Optional[str] = None,
    ):
        self.functions = RECORD_REGISTRY.available_keys() if functions is None else functions
        self.helpers = [cast(Type[RecordHelper], HELPER_REGISTRY.get(h))() for h in helpers]
        self.modality = modality

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
                        modality=self.modality,
                    )
                else:
                    result = dtype.from_file(path)
                logger.debug(f"Created {dtype.__name__} for {path}")
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
            entry = RECORD_REGISTRY.get(name)
            dtype = entry.fn
            assert isinstance(dtype, type)
            assert issubclass(dtype, FileRecord)
            suffixes = set(entry.metadata.get("suffixes", []))
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

    @classmethod
    def precheck_file(cls, path: PathLike, filter_funcs: Iterable[Callable[..., bool]]) -> Optional[Path]:
        path = Path(path)
        if path.is_file() and all(f(path) for f in filter_funcs):
            return path
        return None


def record_iterator(
    files: Iterable[PathLike],
    jobs: Optional[int] = None,
    use_bar: bool = True,
    threads: bool = False,
    record_types: Optional[Iterable[str]] = None,
    helpers: Iterable[str] = [],
    ignore_exceptions: bool = False,
    filters: Iterable[str] = [],
    **kwargs,
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

        filters:
            Iterable of registered names for :class:`RecordFilter`s to use when filtering potential records

    Keyword Args:
        Forwarded to :class:`RecordCreator`

    Returns:
        Iterator of :class:`FileRecord`s
    """
    # build the RecordCreator which determines what FileRecord subclass to use for each file
    filter_funcs: List[Callable[..., bool]] = [FILTER_REGISTRY.get(f)() for f in filters]
    creator = RecordCreator(record_types, helpers, **kwargs)
    logger.debug("Staring record_iterator")
    logger.debug(f"Functions: {creator.functions}")
    logger.debug(f"Helpers: {creator.helpers}")
    logger.debug(f"Filters: {filter_funcs}")

    # build a list of files to check
    # a record creation attempt will only be made on valid files that pass all filters
    Pool = ThreadPoolExecutor if threads else ProcessPoolExecutor
    bar = tqdm(desc="Scanning sources", disable=(not use_bar))
    with Pool(jobs) as p:
        futures = []
        for path in files:
            f = p.submit(creator.precheck_file, path, filter_funcs)
            f.add_done_callback(lambda _: bar.update(1))
            futures.append(f)
        files = [path for f in futures if (path := f.result()) is not None]
    bar.close()

    # create and yield records
    bar = tqdm(desc="Building records", total=len(files), unit="file", disable=(not use_bar))
    with Pool(jobs) as p:
        futures = []
        for path in files:
            f = p.submit(creator, path)
            f.add_done_callback(lambda _: bar.update(1))
            futures.append(f)

        for f in futures:
            if f.exception() and not ignore_exceptions:
                raise cast(Exception, f.exception())
            elif (record := f.result()) and all(f(record) for f in filter_funcs):
                yield record
    bar.close()


T = TypeVar("T", bound=Hashable)
C = TypeVar("C", bound="RecordCollection")


def search_dir(path: PathLike, pattern: str) -> Iterable[Path]:
    path = Path(path)
    if not path.is_dir():
        raise NotADirectoryError(path)
    return path.rglob(pattern)


def iterate_filepaths(
    paths: Iterable[PathLike],
    pattern: str,
    raise_errors: bool = True,
) -> Iterator[Path]:
    for path in paths:
        path = Path(path)
        if path.is_file():
            yield path
        elif path.is_dir():
            for p in search_dir(path, pattern):
                yield p
        elif raise_errors:
            raise FileNotFoundError(path)


R = TypeVar("R", bound=FileRecord)


class RecordCollection(Generic[R]):
    r"""Data stucture for organizing :class:`FileRecord` instances, indexed by various attriburtes."""

    def __init__(self, records: Iterable[R] = set()):
        self.records: Set[R] = set(records)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(length={len(self)})"

    def __len__(self) -> int:
        r"""Returns the total number of contained records"""
        return len(self.records)

    def __iter__(self) -> Iterator[R]:
        for rec in self.records:
            yield rec

    def __contains__(self, x: Any) -> bool:
        return x in self.records

    def add(self, rec: R) -> None:
        return self.records.add(rec)

    def union(self: C, other: C) -> C:
        return self.__class__(self.records.union(other.records))

    def intersection(self: C, other: C) -> C:
        return self.__class__(self.records.intersection(other.records))

    def difference(self: C, other: C) -> C:
        return self.__class__(self.records.difference(other.records))

    def apply(self: C, func: Callable[[R], R]) -> C:
        records = {func(rec) for rec in self.records}
        return self.__class__(records)

    def filter(self: C, func: Callable[[R], bool]) -> C:
        records = {rec for rec in self.records if func(rec)}
        return self.__class__(records)

    def contains_record_type(self, dtype: Type[FileRecord]) -> bool:
        return any(isinstance(rec, dtype) for rec in self)

    @overload
    def group_by(self: C, funcs: Callable[[R], T]) -> Dict[T, C]:
        ...

    @overload
    def group_by(self: C, *funcs: Callable[[R], T]) -> Dict[Tuple[T, ...], C]:
        ...

    def group_by(self: C, *funcs: Callable[[R], Hashable]) -> Dict[Hashable, C]:
        result: Dict[Hashable, C] = {}
        for record in self.records:
            key = tuple(f(record) for f in funcs)
            if len(funcs) == 1:
                key = key[0]
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

    def standardized_filenames(self) -> Iterator[Tuple[Path, R]]:
        counter: Dict[str, int] = {}
        for rec in self:
            path = rec.standardized_filename("id")
            prefix = "_".join(str(path).split("_")[:-1])
            count = counter.get(prefix, 1)
            path = rec.standardized_filename(str(count))
            yield path, rec
            counter[prefix] = count + 1

    def prune_duplicates(self: C, func: Callable[[R], T]) -> C:
        groups = self.group_by(func)
        return self.__class__({next(iter(g)) for g in groups.values()})

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
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
        record_types: Optional[Iterable[str]] = None,
        helpers: Iterable[str] = [],
        filters: Iterable[str] = [],
        **kwargs,
    ) -> C:
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

            record_types:
                List of registered names for :class:`FileRecord` types to try

            helpers:
                Iterable of registered names for :class:`RecordHelper`s to use when creating records

            filters:
                Iterable of registered names for :class:`RecordFilter`s to use when filtering potential records

        Keyword Args:
            Forwarded to :func:`record_iterator`
        """
        return cls.from_files(
            search_dir(path, pattern), jobs, use_bar, threads, record_types, helpers, filters, **kwargs
        )

    @classmethod
    def from_files(
        cls: Type[C],
        files: Iterable[PathLike],
        jobs: Optional[int] = None,
        use_bar: bool = True,
        threads: bool = False,
        record_types: Optional[Iterable[str]] = None,
        helpers: Iterable[str] = [],
        filters: Iterable[str] = [],
        **kwargs,
    ) -> C:
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

            record_types:
                List of registered names for :class:`FileRecord` types to try

            helpers:
                Iterable of registered names for :class:`RecordHelper`s to use when creating records

            filters:
                Iterable of registered names for :class:`RecordFilter`s to use when filtering potential records

        Keyword Args:
            Forwarded to :func:`record_iterator`
        """
        collection = cls(
            record_iterator(files, jobs, use_bar, threads, record_types, helpers, filters=filters, **kwargs)
        )
        return collection

    @classmethod
    def create(
        cls: Type[C],
        paths: Iterable[PathLike],
        pattern: str = "*",
        jobs: Optional[int] = None,
        use_bar: bool = True,
        threads: bool = False,
        record_types: Optional[Iterable[str]] = None,
        helpers: Iterable[str] = [],
        filters: Iterable[str] = [],
        **kwargs,
    ) -> C:
        r"""Create a :class:`RecordCollection` from a list of paths, either files or directories.
        If a :class:`FileRecord` cannot be created for a file, that file is silently excluded
        from the collection.

        Args:
            paths:
                List of paths to create records from

            pattern:
                Glob pattern for matching files

            jobs:
                Number of parallel jobs to use

            use_bar:
                If ``False``, don't show a tqdm progress bar

            threads:
                If ``True``, use a :class:`ThreadPoolExecutor`. Otherwise, use a :class:`ProcessPoolExecutor`

            record_types:
                List of registered names for :class:`FileRecord` types to try

            helpers:
                Iterable of registered names for :class:`RecordHelper`s to use when creating records

            filters:
                Iterable of registered names for :class:`RecordFilter`s to use when filtering potential records

        Keyword Args:
            Forwarded to :func:`record_iterator`
        """
        paths = iterate_filepaths(paths, pattern)
        return cls.from_files(paths, jobs, use_bar, threads, record_types, helpers, filters=filters, **kwargs)

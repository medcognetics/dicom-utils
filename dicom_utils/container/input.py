#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial, reduce
from os import PathLike
from pathlib import Path
from typing import Callable, Dict, Generic, Hashable, Iterable, Iterator, Optional, Tuple, Type, TypeVar, Union, cast

from registry import Registry

from .collection import RecordCollection
from .group import GROUP_REGISTRY
from .record import (
    RECORD_REGISTRY,
    DicomFileRecord,
    FileRecord,
    SupportsPatientID,
    SupportsStudyDate,
    SupportsStudyUID,
)


NAME_REGISTRY = Registry("names", bound=Type[Callable[[Hashable, RecordCollection, int, int], str]])
K = TypeVar("K", bound=Hashable)


class CaseRenamer(ABC, Generic[K]):
    @abstractmethod
    def __call__(self, key: K, collection: RecordCollection, index: int, total: int) -> str:
        raise NotImplementedError

    @classmethod
    def num_leading_zeros(cls, total: int) -> int:
        return len(str(total))

    @classmethod
    def add_leading_zeros(cls, index: int, total: int) -> str:
        return str(index).zfill(cls.num_leading_zeros(total))


@NAME_REGISTRY(name="key")
@dataclass
class UseKeyRenamer(CaseRenamer):
    prefix: str = ""

    def __call__(self, key: Hashable, collection: RecordCollection, index: int, total: int) -> str:
        key = str(key) if key is not None else "???-{index}"
        return f"{self.prefix}{key}"


@NAME_REGISTRY(name="consecutive")
@dataclass
class ConsecutiveNamer(CaseRenamer[K]):
    prefix: str = "Case-"
    start: int = 1

    def __call__(self, key: K, collection: RecordCollection, index: int, total: int) -> str:
        return f"{self.prefix}{self.add_leading_zeros(index, total)}"


@NAME_REGISTRY(name="patient-id")
@dataclass
class PatientIDNamer(CaseRenamer[Optional[str]]):
    prefix: str = "Patient-"

    def __call__(self, key: Optional[str], collection: RecordCollection, index: int, total: int) -> str:
        pids = {pid for rec in collection if isinstance(rec, SupportsPatientID) and (pid := rec.PatientID) is not None}
        if len(pids) != 1:
            raise ValueError("Expected 1 PatientID, but found {len(pids)}")
        pid = next(iter(pids))
        return f"{self.prefix}{pid}"


@NAME_REGISTRY(name="study-date")
@dataclass
class StudyDateNamer(CaseRenamer):
    prefix: str = "Date-"
    year_only: bool = False

    def __call__(self, key: Optional[str], collection: RecordCollection, index: int, total: int) -> str:
        dates = {
            date
            for rec in collection
            if isinstance(rec, SupportsStudyDate)
            and (date := (rec.StudyYear if self.year_only else rec.StudyDate)) is not None
        }
        if len(dates) != 1:
            raise ValueError("Expected 1 StudyDate, but found {len(dates)}")
        date = next(iter(dates))
        return f"{self.prefix}{date}"


@NAME_REGISTRY(name="study-uid")
@dataclass
class StudyIDNamer(CaseRenamer[Optional[str]]):
    prefix: str = "Study-"

    def __call__(self, key: Optional[str], collection: RecordCollection, index: int, total: int) -> str:
        uids = {
            uid for rec in collection if isinstance(rec, SupportsStudyUID) and (uid := rec.StudyInstanceUID) is not None
        }
        if len(uids) != 1:
            raise ValueError("Expected 1 StudyInstanceUID, but found {len(pids)}")
        uid = next(iter(uids))
        return f"{self.prefix}{uid}"


class Input:
    r"""Input pipeline for discovering files, creating :class:`FileRecord`s, grouping records, and naming
    each group. :class:`Input` is an iterable over the deterministically sorted record groups and their
    corresponding group name.

    Args:
        sources:
            A directory or iterable of directories from which to read

        records:
            An iterable of names for :class:`FileRecord` subclasses registered in ``RECORD_REGISTRY``.
            Defaults to all registered :class:`FileRecord` subclasses.

        groups:
            An iterable of names for grouping functions registered in ``GROUP_REGISTRY``.
            Defaults to grouping by StudyInstanceUID.

        helpers:
            An iterable of names for :class:`RecordHelper` subclasses registered in ``HELPER_REGISTRY``.
            By default no helpers will be used.

        filters:
            An iterable of names for :class:`RecordFilter` subclasses registered in ``FILTER_REGISTRY``.
            By default no filters will be used.

    """

    def __init__(
        self,
        sources: Union[PathLike, Iterable[PathLike]],
        records: Optional[Iterable[str]] = None,
        groups: Iterable[str] = ["patient-id", "study-date", "study-uid"],
        helpers: Iterable[str] = [],
        namers: Iterable[str] = ["patient-id", "study-date", "study-uid"],
        filters: Iterable[str] = [],
        require_dicom: bool = False,
        **kwargs,
    ):
        if records is None:
            self.records = RECORD_REGISTRY.available_keys()
        else:
            self.records = records
        self.groups = [GROUP_REGISTRY.get(g) for g in groups]
        self.namers = [NAME_REGISTRY.get(n)() for n in namers]
        if len(self.namers) != len(self.groups):
            raise ValueError("Number of namers {namers} should match number of groups {groups}")

        # scan sources and build a RecordCollection with every valid file found
        sources = [Path(sources)] if isinstance(sources, PathLike) else [Path(p) for p in sources]
        scan_source = partial(
            RecordCollection.from_dir, record_types=self.records, helpers=helpers, filters=filters, **kwargs
        )
        collection = reduce(
            lambda c1, c2: c1.union(c2),
            (scan_source(s) for s in sources),
        )

        # apply groupers to generate a dict of key -> group pairs
        grouped_collections = {
            key: grouped
            for key, grouped in collection.group_by(*self.groups).items()
            if not require_dicom or grouped.contains_record_type(DicomFileRecord)
        }

        self.cases: Dict[Tuple[str, ...], RecordCollection] = {}
        for i, group_key in enumerate(sorted(grouped_collections.keys())):
            group = grouped_collections[group_key]
            group_key = (group_key,) if not isinstance(group_key, tuple) else group_key
            key = tuple(namer(k, group, i + 1, len(grouped_collections)) for namer, k in zip(self.namers, group_key))
            self.cases[key] = group

    @property
    def group_fn(self) -> Callable[[FileRecord], Hashable]:
        return cast(Callable[[FileRecord], Hashable], self.groups[0])

    def __iter__(self) -> Iterator[Tuple[Tuple[str, ...], RecordCollection]]:
        r"""Iterates over pairs of named groups and the :class:`RecordCollection` containing that group."""
        for k, v in self.cases.items():
            yield k, v

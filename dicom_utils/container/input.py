#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from os import PathLike
from pathlib import Path
from typing import Callable, Dict, Hashable, Iterable, Iterator, Optional, Tuple, Type, Union, cast

from registry import Registry

from .collection import RecordCollection
from .group import GROUP_REGISTRY
from .record import RECORD_REGISTRY, DicomFileRecord, FileRecord, SupportsPatientID, SupportsStudyDate


NAME_REGISTRY = Registry("names", bound=Type[Callable[[RecordCollection, int, int], str]])


class CaseRenamer(ABC):
    @abstractmethod
    def __call__(self, collection: RecordCollection, index: int, total: int) -> str:
        raise NotImplementedError

    @classmethod
    def num_leading_zeros(cls, total: int) -> int:
        return len(str(total))

    @classmethod
    def add_leading_zeros(cls, index: int, total: int) -> str:
        return str(index).zfill(cls.num_leading_zeros(total))


@NAME_REGISTRY(name="consecutive")
@dataclass
class ConsecutiveNamer(CaseRenamer):
    prefix: str = "Case-"
    start: int = 1

    def __call__(self, collection: RecordCollection, index: int, total: int) -> str:
        return f"{self.prefix}{self.add_leading_zeros(index, total)}"


@NAME_REGISTRY(name="patient")
@dataclass
class PatientIDNamer(CaseRenamer):
    prefix: str = "Patient-"

    def __call__(self, collection: RecordCollection, index: int, total: int) -> str:
        patient_ids = {
            pid for rec in collection if isinstance(rec, SupportsPatientID) and (pid := rec.PatientID) is not None
        }
        if len(patient_ids) != 1:
            raise ValueError("Expected 1 PatientID, but found {len(patient_ids)}")
        pid = next(iter(patient_ids))
        return f"{self.prefix}{pid}"


@NAME_REGISTRY(name="study-date")
@dataclass
class StudyDateNamer(CaseRenamer):
    prefix: str = "Date-"
    year_only: bool = False

    def __call__(self, collection: RecordCollection, index: int, total: int) -> str:
        dates = {
            date
            for rec in collection
            if isinstance(rec, SupportsStudyDate)
            and (date := (rec.StudyYear if self.year_only else rec.StudyDate)) is not None
        }
        if len(dates) != 1:
            raise ValueError("Expected 1 date, but found {len(dates)}")
        date = next(iter(dates))
        return f"{self.prefix}{date}"


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

        prefix:
            A prefix to use when naming each case.

        start:
            Value at which to start numbering cases.
    """

    def __init__(
        self,
        sources: Union[PathLike, Iterable[PathLike]],
        records: Optional[Iterable[str]] = None,
        groups: Iterable[str] = ["patient-id", "study-uid"],
        helpers: Iterable[str] = ["case"],
        namer: str = "consecutive",
        start: int = 1,
        require_dicom: bool = True,
        **kwargs,
    ):
        if records is None:
            self.records = RECORD_REGISTRY.available_keys()
        else:
            self.records = records
        self.groups = [GROUP_REGISTRY.get(g) for g in groups]
        self.namer = NAME_REGISTRY.get(namer)()
        self.start = start

        # scan sources and build a RecordCollection with every valid file found
        sources = [Path(sources)] if isinstance(sources, PathLike) else [Path(p) for p in sources]
        collection = reduce(
            lambda c1, c2: c1.union(c2),
            (RecordCollection.from_dir(s, record_types=self.records, helpers=helpers, **kwargs) for s in sources),
        )

        # apply groupers to generate a dict of key -> group pairs
        grouped_collections = {
            key: grouped
            for key, grouped in collection.group_by(*self.groups).items()
            if not require_dicom or grouped.contains_record_type(DicomFileRecord)
        }

        self.cases: Dict[str, RecordCollection] = {}
        for i, k in enumerate(sorted(grouped_collections.keys())):
            group = grouped_collections[k]
            name = self.namer(group, i + 1, len(grouped_collections))
            self.cases[name] = group

    @property
    def group_fn(self) -> Callable[[FileRecord], Hashable]:
        return cast(Callable[[FileRecord], Hashable], self.groups[0])

    def __iter__(self) -> Iterator[Tuple[str, RecordCollection]]:
        r"""Iterates over pairs of named groups and the :class:`RecordCollection` containing that group."""
        for k, v in self.cases.items():
            yield k, v

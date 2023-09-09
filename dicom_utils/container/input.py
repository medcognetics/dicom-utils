#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from registry import Registry, bind_relevant_kwargs

from .collection import RecordCollection
from .group import Grouper
from .record import RECORD_REGISTRY, SupportsPatientID, SupportsStudyDate, SupportsStudyUID


logger = logging.getLogger(__name__)
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
        pid = self.get_patient_id(collection)
        return f"{self.prefix}{pid}" if pid else ""

    @classmethod
    def get_patient_id(cls, collection: RecordCollection) -> Optional[str]:
        pids = {pid for rec in collection if isinstance(rec, SupportsPatientID) and (pid := rec.PatientID) is not None}
        pid = next(iter(pids)) if len(pids) == 1 else None
        # Replace slashes with underscores because they are not allowed in filenames
        if pid is not None:
            pid = pid.replace("/", "_")
        return pid


@NAME_REGISTRY(name="study-date")
@dataclass
class StudyDateNamer(CaseRenamer):
    prefix: str = "Date-"
    year_only: bool = False

    def __call__(self, key: Optional[str], collection: RecordCollection, index: int, total: int) -> str:
        date = self.get_study_date(collection, self.year_only)
        return f"{self.prefix}{date}" if date is not None else ""

    @classmethod
    def get_study_date(cls, collection: RecordCollection, year_only: bool = False) -> Optional[str]:
        dates = {
            str(date)
            for rec in collection
            if isinstance(rec, SupportsStudyDate)
            and (date := (rec.StudyYear if year_only else rec.StudyDate)) is not None
        }
        return next(iter(dates)) if len(dates) == 1 else None


@NAME_REGISTRY(name="study-uid")
@dataclass
class StudyIDNamer(CaseRenamer[Optional[str]]):
    prefix: str = "Study-"
    truncate: Optional[int] = 4
    strip_period: bool = True
    patient_namer: Optional[str] = None
    date_namer: Optional[str] = None

    _patient_namer: Optional[PatientIDNamer] = field(default=None, init=False)
    _date_namer: Optional[StudyDateNamer] = field(default=None, init=False)

    def __post_init__(self):
        if self.patient_namer:
            self._patient_namer = cast(
                PatientIDNamer,
                NAME_REGISTRY.get(self.patient_namer).instantiate_with_metadata().fn,
            )
            assert isinstance(self._patient_namer, PatientIDNamer), type(self._patient_namer)
        if self.date_namer:
            self._date_namer = cast(
                StudyDateNamer,
                NAME_REGISTRY.get(self.date_namer).instantiate_with_metadata().fn,
            )
            assert isinstance(self._date_namer, StudyDateNamer), type(self._date_namer)

    def __call__(self, key: Optional[str], collection: RecordCollection, index: int, total: int) -> str:
        uid = self.get_study_uid(collection)
        if not uid:
            return ""

        uid = self._strip(uid)
        uid = self._truncate(uid)
        pid = self._call_patient_namer(key, collection, index, total) if self._patient_namer is not None else ""
        date = self._call_date_namer(key, collection, index, total) if self._date_namer is not None else ""

        parts = [s for s in (pid, date, f"{self.prefix}{uid}") if s]
        return "_".join(parts)

    @classmethod
    def get_study_uid(cls, collection: RecordCollection) -> Optional[str]:
        uids = {
            uid for rec in collection if isinstance(rec, SupportsStudyUID) and (uid := rec.StudyInstanceUID) is not None
        }
        return next(iter(uids)) if len(uids) == 1 else None

    def _truncate(self, uid: str) -> str:
        if self.truncate is not None:
            assert self.truncate > 0
            uid = uid[-self.truncate :]
        return uid

    def _strip(self, uid: str) -> str:
        if self.strip_period:
            uid = uid.replace(".", "")
        return uid

    def _call_patient_namer(self, key: Optional[str], collection: RecordCollection, index: int, total: int) -> str:
        assert self._patient_namer is not None
        return self._patient_namer(key, collection, index, total)

    def _call_date_namer(self, key: Optional[str], collection: RecordCollection, index: int, total: int) -> str:
        assert self._date_namer is not None
        return self._date_namer(key, collection, index, total)


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
        groups: Iterable[str] = ["patient-id", "study-uid"],
        helpers: Iterable[str] = [],
        namers: Iterable[str] = ["patient-id", "study-uid"],
        filters: Iterable[str] = [],
        **kwargs,
    ):
        if records is None:
            self.records = RECORD_REGISTRY.available_keys()
        else:
            self.records = records
        namers = list(namers)
        groups = list(groups)
        helpers = list(helpers)

        self.grouper = bind_relevant_kwargs(Grouper, groups=groups, helpers=helpers, **kwargs)()
        self.namers = [NAME_REGISTRY.get(n).instantiate_with_metadata() for n in namers]
        if len(namers) != len(groups):
            raise ValueError("Number of namers {namers} should match number of groups {groups}")

        # scan sources and build a RecordCollection with every valid file found
        sources = (Path(p) for p in ([sources] if isinstance(sources, PathLike) else sources))
        collection = RecordCollection.create(
            sources, record_types=self.records, helpers=helpers, filters=filters, **kwargs
        )

        # apply groupers to generate a dict of key -> group pairs
        grouped_collections = self.grouper(collection)

        # apply namers to generate a dict of (group name) -> group pairs
        self.cases: Dict[Tuple[str, ...], RecordCollection] = {}
        for i, group_key in enumerate(sorted(grouped_collections.keys(), key=cast(Any, self.sort_key))):
            group = grouped_collections[group_key]
            group_key = (group_key,) if not isinstance(group_key, tuple) else group_key
            key = tuple(namer(k, group, i + 1, len(grouped_collections)) for namer, k in zip(self.namers, group_key))
            self.cases[key] = group

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[Tuple[Tuple[str, ...], RecordCollection]]:
        r"""Iterates over pairs of named groups and the :class:`RecordCollection` containing that group."""
        for k, v in self.cases.items():
            yield k, v

    @staticmethod
    def sort_key(key: Optional[Tuple[Hashable, ...]]) -> Tuple[str, ...]:
        if key is None:
            return ("None",)
        return tuple(str(x) for x in key)

#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging
import warnings
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Final, Hashable, Iterator, Optional, Sequence, Tuple, TypeVar, cast

from registry import Registry
from tqdm_multiprocessing import ConcurrentMapper

from .collection import CollectionHelper, RecordCollection, apply_helpers, get_bar_description_suffix

# Type checking fails when dataclass attr name matches a type alias.
# Import types under a different alias
from .helpers import StudyUID
from .record import HELPER_REGISTRY, FileRecord


H = TypeVar("H", bound=Hashable)


GroupFunction = Callable[[FileRecord], Hashable]
GROUP_REGISTRY = Registry("group", bound=Callable[..., Hashable])
Key = Tuple[Hashable, ...]
GroupDict = Dict[Key, RecordCollection]
COMPLETE_COLLECTION_INDEX: Final[int] = 0
FIRST_GROUP_INDEX: Final[int] = 1

logger = logging.getLogger(__name__)


@GROUP_REGISTRY(name="study-uid")
def group_by_study_instance_uid(rec: FileRecord) -> Optional[StudyUID]:
    return getattr(rec, "StudyInstanceUID", None)


@GROUP_REGISTRY(name="parent")
def group_by_parent(rec: FileRecord, level: int = 0) -> Path:
    return rec.path.parents[level]


for i in range(3):
    GROUP_REGISTRY(partial(group_by_parent, level=i + 1), name=f"parent-{i+1}")


@GROUP_REGISTRY(name="patient-id")
def group_by_patient_id(rec: FileRecord) -> Optional[str]:
    return getattr(rec, "PatientID", None)


@GROUP_REGISTRY(name="study-date")
def group_by_study_date(rec: FileRecord) -> Optional[str]:
    return getattr(rec, "StudyDate", None)


@dataclass
class Grouper:
    groups: Sequence[str]
    helpers: Sequence[str]
    threads: bool = False
    jobs: Optional[int] = None
    chunksize: int = 8
    use_bar: bool = True
    timeout: Optional[int] = None

    _group_fns: Sequence[GroupFunction] = field(init=False)
    _helper_fns: Sequence[CollectionHelper] = field(init=False)

    def __post_init__(self):
        if not self.groups:
            raise ValueError("`groups` cannot be empty")

        self._group_fns = [GROUP_REGISTRY.get(g).instantiate_with_metadata().fn for g in self.groups]
        self._helper_fns = list(self._build_collection_helpers())
        logger.info(f"Collection helpers: {self._helper_fns}")

        # Using processes seems to result in deadlocks. This is a workaround.
        if not self.threads:
            self.threads = True

    def _build_collection_helpers(self) -> Iterator[CollectionHelper]:
        for h in self.helpers:
            helper = HELPER_REGISTRY.get(h).instantiate_with_metadata().fn
            # isolate CollectionHelpers
            if isinstance(helper, CollectionHelper):
                yield helper
            else:
                logger.debug(f"Ignoring non-RecordHelper `{h}` of type {type(h)}")

    def __call__(self, collection: RecordCollection) -> Dict[Hashable, RecordCollection]:
        start_len = len(collection)

        # Apply helpers at the entire collection level
        collection = apply_helpers(collection, self._helper_fns, index=COMPLETE_COLLECTION_INDEX)

        result: Dict[Key, RecordCollection] = {tuple(): collection}

        with ConcurrentMapper(self.threads, self.jobs, chunksize=self.chunksize, timeout=self.timeout) as mapper:
            for i, group_fn in list(enumerate(self._group_fns)):
                # run the group function
                total = sum(len(v) for v in result.values())
                mapper.create_bar(
                    desc=f"Running grouper {self.groups[i]} ({get_bar_description_suffix(self.jobs, self.threads)})",
                    disable=(not self.use_bar),
                    leave=True,
                    total=total,
                )
                _result: Dict[Key, RecordCollection] = {}
                needs_grouping: Iterator[Tuple[Key, FileRecord]] = (
                    (key, record) for key, collection in result.items() for record in collection
                )
                # TODO: This can deadlock. Only seems to happen on Optimam full dataset.
                for key, record in mapper(self._group, needs_grouping, group_fn=group_fn):
                    _result.setdefault(key, RecordCollection()).add(record)
                result = _result
                mapper.close_bar()

                # run helpers for this stage in the grouping process
                mapper.create_bar(
                    desc=f"Running group helpers ({get_bar_description_suffix(self.jobs, self.threads)})",
                    disable=(not self.use_bar),
                    leave=True,
                    total=len(result),
                )
                mapped = mapper(self._apply_helpers, list(result.items()), index=i + 1)
                result = {k: v for k, v in mapped}
                mapper.close_bar()

        end_len = sum(len(collection) for collection in result.values())
        if start_len != end_len:
            warnings.warn(
                f"Grouping began with {start_len} records and ended with {end_len} records. "
                "This may be expected behavior depending on what helpers are being used."
            )
        return cast(Dict[Hashable, RecordCollection], result)

    @classmethod
    def _group(cls, inp: Tuple[Key, FileRecord], group_fn: GroupFunction) -> Tuple[Key, FileRecord]:
        key, entry = inp
        key = key + (group_fn(entry),)
        return key, entry

    def _apply_helpers(self, inp: Tuple[Key, RecordCollection], index: int) -> Tuple[Key, RecordCollection]:
        k, col = inp
        return k, apply_helpers(col, self._helper_fns, index=index)

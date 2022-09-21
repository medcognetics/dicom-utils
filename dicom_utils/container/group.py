#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging
import warnings
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Hashable, Iterator, List, Optional, Sequence, Tuple, TypeVar, cast

from registry import Registry
from tqdm_multiprocessing import ConcurrentMapper

from .collection import CollectionHelper, RecordCollection, apply_helpers

# Type checking fails when dataclass attr name matches a type alias.
# Import types under a different alias
from .helpers import StudyUID
from .record import HELPER_REGISTRY, FileRecord


H = TypeVar("H", bound=Hashable)


GroupFunction = Callable[[FileRecord], Hashable]
GROUP_REGISTRY = Registry("group", bound=Callable[..., Hashable])
Key = Tuple[Hashable, ...]
GroupDict = Dict[Key, RecordCollection]

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

    _group_fns: Sequence[GroupFunction] = field(init=False)
    _helper_fns: Sequence[CollectionHelper] = field(init=False)

    def __post_init__(self):
        if not self.groups:
            raise ValueError("`groups` cannot be empty")

        self._group_fns = [GROUP_REGISTRY.get(g).instantiate_with_metadata().fn for g in self.groups]
        self._helper_fns = list(self._build_collection_helpers())
        logger.info(f"Collection helpers: {self._helper_fns}")

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
        result: Dict[Key, RecordCollection] = {tuple(): collection}

        with ConcurrentMapper(self.threads, self.jobs, chunksize=self.chunksize) as mapper:
            for i, group_fn in list(enumerate(self._group_fns)):
                # run the group function
                mapper.create_bar(
                    desc=f"Running grouper {self.groups[i]}", disable=(not self.use_bar), leave=False, total=len(result)
                )
                mapped = mapper(self._group, list(result.items()), group_fn=group_fn)
                result = {k: v for group_result in mapped for k, v in group_result}
                mapper.close_bar()

                # run helpers for this stage in the grouping process
                mapper.create_bar(
                    desc="Running group helpers", disable=(not self.use_bar), leave=False, total=len(result)
                )
                mapped = mapper(self._apply_helpers, list(result.items()), index=i)
                result = {k: v for k, v in mapped}
                mapper.close_bar()

        end_len = sum(len(collection) for collection in result.values())
        if start_len != end_len:
            warnings.warn(
                "Grouping began with {start_len} records and ended with {end_len} records. "
                "This may be expected behavior depending on what helpers are being used."
            )
        return cast(Dict[Hashable, RecordCollection], result)

    @classmethod
    def _group(cls, inp: Tuple[Key, RecordCollection], group_fn: GroupFunction) -> List[Tuple[Key, RecordCollection]]:
        key, entry = inp
        result: List[Tuple[Key, RecordCollection]] = []
        for subkey, subgroup in entry.group_by(group_fn).items():
            new_key = key + (subkey,)
            result.append((new_key, subgroup))
        return result

    def _apply_helpers(self, inp: Tuple[Key, RecordCollection], index: int) -> Tuple[Key, RecordCollection]:
        k, col = inp
        return k, apply_helpers(col, self._helper_fns, index=index)

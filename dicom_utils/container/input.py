#!/usr/bin/env python
# -*- coding: utf-8 -*-
from functools import reduce
from os import PathLike
from pathlib import Path
from typing import Callable, Dict, Hashable, Iterable, Iterator, Optional, Tuple, Union, cast

from .collection import RecordCollection
from .group import GROUP_REGISTRY
from .record import RECORD_REGISTRY, FileRecord


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
        groups: Iterable[str] = ["study-uid"],
        helpers: Iterable[str] = [],
        prefix: str = "Case-",
        start: int = 1,
        **kwargs,
    ):
        if records is None:
            self.records = RECORD_REGISTRY.available_keys()
        else:
            self.records = records
        self.groups = [GROUP_REGISTRY.get(g) for g in groups]
        self.prefix = prefix
        self.start = start

        sources = [Path(sources)] if isinstance(sources, PathLike) else [Path(p) for p in sources]
        collection = reduce(
            lambda c1, c2: c1.union(c2),
            (RecordCollection.from_dir(s, record_types=self.records, helpers=helpers, **kwargs) for s in sources),
        )

        self.cases = self.to_ordered_collections(collection.group_by(self.group_fn).values())

    @property
    def group_fn(self) -> Callable[[FileRecord], Hashable]:
        return cast(Callable[[FileRecord], Hashable], self.groups[0])

    def __iter__(self) -> Iterator[Tuple[str, RecordCollection]]:
        r"""Iterates over pairs of named groups and the :class:`RecordCollection` containing that group."""
        for k, v in self.cases.items():
            yield k, v

    def to_ordered_collections(self, collections: Iterable[RecordCollection]) -> Dict[str, RecordCollection]:
        collections = list(collections)
        num_leading_zeros = len(str(len(collections)))

        def get_postfix(i):
            return str(i + self.start).zfill(num_leading_zeros)

        return {
            f"{self.prefix}{get_postfix(i)}": col
            for i, col in enumerate(sorted(collections, key=lambda c: str(c.sort_key)))
        }

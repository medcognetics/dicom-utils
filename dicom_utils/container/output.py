#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
from abc import ABC, abstractmethod
from functools import partial
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, Final, Hashable, Iterable, Optional, Set, Union, cast

from tqdm import tqdm

from .collection import RecordCollection
from .input import Input
from .record import FileRecord, MammogramFileRecord
from .registry import Registry


YEAR_RE: Final = re.compile(r"\d{4}")

OUTPUT_REGISTRY = Registry("output")


class Output(ABC):
    r"""An :class:`Output` implements logic to output a some data product from an :class:`Input`
    or the data products produced by another :class:`Output`.

    Args:
        path:
            Directory the outputs will be written.

        record_filter:
            Filter function to exclude individual :class:`FileRecord`s from the output.

        collection_filter:
            Filter function to exclude entire :class:`RecordCollection`s from the output.
    """

    def __init__(
        self,
        path: PathLike,
        record_filter: Optional[Callable[[FileRecord], bool]] = None,
        collection_filter: Optional[Callable[[RecordCollection], bool]] = None,
        use_bar: bool = True,
    ):
        self.path = Path(path)
        self.path.mkdir(exist_ok=True, parents=True)
        self.record_filter = record_filter
        self.collection_filter = collection_filter
        self.use_bar = use_bar

    def __call__(self, inp: Union[Input, Dict[str, RecordCollection]]) -> Dict[str, RecordCollection]:
        result: Dict[str, RecordCollection] = {}
        bar = tqdm(
            inp if isinstance(inp, Input) else inp.items(),
            leave=False,
            disable=not self.use_bar,
        )
        for name, collection in bar:
            if self.collection_filter is not None and not self.collection_filter(collection):
                continue
            if self.record_filter is not None:
                collection = collection.filter(self.record_filter)
            dest = Path(self.path, name)
            dest.mkdir(exist_ok=True, parents=True)
            written = self.write(collection, dest)
            result[name] = written
            if isinstance(inp, Input):
                key = inp.group_fn(next(iter(collection)))
                self.write_metadata(name, key, collection, dest)
        return result

    @abstractmethod
    def write(self, collection: RecordCollection, dest: Path) -> RecordCollection:
        ...

    def write_metadata(
        self,
        name: str,
        key: Hashable,
        collection: RecordCollection,
        dest: Path,
        indent: int = 4,
    ) -> None:
        path = Path(dest, "record_collection.json")
        metadata = collection.to_dict()
        metadata["group"] = key
        metadata["name"] = name
        with open(path, "w") as f:
            json.dump(metadata, f, sort_keys=True, indent=indent, default=lambda o: str(o))


class SymlinkFileOutput(Output):
    def write(self, collection: RecordCollection, dest: Path) -> RecordCollection:
        assert dest.is_dir()
        result = RecordCollection()
        for name, rec in collection.standardized_filenames():
            filepath = Path(dest, name)
            rec = rec.to_symlink(filepath, overwrite=True)
            result.add(rec)
        return result


@OUTPUT_REGISTRY(name="json", subdir="cases")
class JsonFileOutput(Output):
    def write(self, collection: RecordCollection, dest: Path) -> RecordCollection:
        assert dest.is_dir()
        result = RecordCollection()
        metadata: Dict[str, Any] = {}
        metadata["StudyInstanceUID"]
        return result


@OUTPUT_REGISTRY(name="longitudinal", subdir="cases", derived=True)
class LongitudinalPointerOutput(Output):
    def __call__(self, inp: Union[Input, Dict[str, RecordCollection]]) -> Dict[str, RecordCollection]:
        assert isinstance(inp, dict)
        # build lookup for years and related cases
        year_lookup = self.create_year_lookup(inp)
        association_lookup = self.create_association_lookup(inp)
        reverse_association_lookup = {name: group for group, names in association_lookup.items() for name in names}

        result: Dict[str, RecordCollection] = {}
        bar = tqdm(
            inp.items(),
            leave=False,
            disable=not self.use_bar,
        )
        for name, collection in bar:
            if self.collection_filter is not None and not self.collection_filter(collection):
                continue
            if self.record_filter is not None:
                collection = collection.filter(self.record_filter)

            dest = Path(self.path, name)
            group = reverse_association_lookup.get(name, None)
            if group is None:
                continue
            associated = {x for x in association_lookup[group] if x != name}

            for a in associated:
                year = year_lookup[a]
                reference_year = year_lookup.get(name, year)
                if year > reference_year:
                    link = Path(dest, "later", a)
                elif year < reference_year:
                    link = Path(dest, "prior", a)
                else:
                    link = Path(dest, "related", a)

                link.parent.mkdir(exist_ok=True)
                source = Path(self.path, a)
                written = self.write(source, link)
                result[name] = written

        return result

    def create_year_lookup(self, inp: Dict[str, RecordCollection]) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for name, collection in inp.items():
            years = {year for rec in collection if (year := getattr(rec, "year", None)) is not None}
            if years:
                result[name] = next(iter(years))
        return result

    def create_association_lookup(self, inp: Dict[str, RecordCollection]) -> Dict[str, Set[str]]:
        result: Dict[str, Set[str]] = {}
        for name, collection in inp.items():
            patient_ids = {year for rec in collection if (year := getattr(rec, "PatientID", None)) is not None}
            if patient_ids:
                result.setdefault(next(iter(patient_ids)), set()).add(name)
        return result

    def write(self, source: Path, dest: Path) -> RecordCollection:
        assert source.is_dir()
        assert dest.parent.is_dir()
        relpath = Path(*FileRecord(source).relative_to(FileRecord(dest)).path.parts)
        dest.unlink(missing_ok=True)
        dest.symlink_to(relpath, target_is_directory=True)
        return RecordCollection([FileRecord(dest)])


def is_complete_case(c: RecordCollection) -> bool:
    mammograms = c.filter(lambda rec: isinstance(rec, MammogramFileRecord))
    assert all(isinstance(rec, MammogramFileRecord) for rec in mammograms)
    mammograms = cast(Iterable[MammogramFileRecord], mammograms)
    return MammogramFileRecord.is_complete_mammo_case(mammograms)


def is_mammogram_case(c: RecordCollection) -> bool:
    return any(isinstance(rec, MammogramFileRecord) for rec in c)


OUTPUT_REGISTRY(
    SymlinkFileOutput,
    name="symlink-cases",
    subdir="cases",
)
OUTPUT_REGISTRY(
    partial(SymlinkFileOutput, collection_filter=is_mammogram_case),
    name="symlink-mammograms",
    subdir="mammograms",
)
OUTPUT_REGISTRY(
    partial(SymlinkFileOutput, collection_filter=is_complete_case),
    name="symlink-complete-mammograms",
    subdir="complete-mammograms",
)

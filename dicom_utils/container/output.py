#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
from abc import ABC, abstractclassmethod
from functools import partial
from os import PathLike
from pathlib import Path
from typing import Callable, Dict, Final, Hashable, Iterable, List, Optional, Set, Tuple, Union, cast

from registry import Registry
from tqdm import tqdm
from tqdm_multiprocessing import ConcurrentMapper

from ..types import MammogramType
from .collection import RecordCollection
from .input import Input
from .record import FileRecord, MammogramFileRecord


YEAR_RE: Final = re.compile(r"\d{4}")

OUTPUT_REGISTRY = Registry("output", bound=Callable[..., "Output"])


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
        threads: bool = False,
        jobs: Optional[int] = None,
    ):
        self.path = Path(path)
        self.path.mkdir(exist_ok=True, parents=True)
        self.record_filter = record_filter
        self.collection_filter = collection_filter
        self.use_bar = use_bar
        self.threads = threads
        self.jobs = jobs

    def __call__(self, inp: Union[Input, Dict[Tuple[str, ...], RecordCollection]]) -> Dict[str, RecordCollection]:
        result: Dict[str, RecordCollection] = {}
        iterable = inp if isinstance(inp, Input) else inp.items()
        with ConcurrentMapper(self.threads, self.jobs, chunksize=8) as mapper:
            mapper.create_bar(total=len(iterable), desc=f"Writing {self.__class__.__name__}")
            it = mapper(
                self._process,
                iterable,
                path=self.path,
                record_filter=self.record_filter,
                collection_filter=self.collection_filter,
                group_fn=inp.group_fn if isinstance(inp, Input) else None,
            )
            for item in it:
                if item is not None:
                    name, written = item
                    result[name] = written
        return result

    @classmethod
    def _process(
        cls,
        kc,
        path: Path,
        record_filter: Optional[Callable[[FileRecord], bool]] = None,
        collection_filter: Optional[Callable[[RecordCollection], bool]] = None,
        group_fn: Optional[Callable] = None,
    ):
        key, collection = kc
        name = str(Path(*key))
        if collection_filter is not None and not collection_filter(collection):
            return
        if record_filter is not None:
            collection = collection.filter(record_filter)
        dest = Path(path, name)
        dest.mkdir(exist_ok=True, parents=True)
        # pyright fails to recognize this is an abstract classmethod
        written = cls.write(collection, dest)  # type: ignore
        if group_fn is not None:
            key = group_fn(next(iter(collection)))
            cls.write_metadata(name, key, collection, dest)
        return name, written

    @property
    def tqdm(self) -> Callable:
        return partial(tqdm, leave=False, disable=not self.use_bar)

    @abstractclassmethod
    def write(cls, collection: RecordCollection, dest: Path) -> RecordCollection:
        ...

    @classmethod
    def write_metadata(
        cls,
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
            json.dump(metadata, f, sort_keys=True, indent=indent, default=str)


class SymlinkFileOutput(Output):
    @classmethod
    def write(cls, collection: RecordCollection, dest: Path) -> RecordCollection:
        assert dest.is_dir()
        result = RecordCollection()
        for name, rec in collection.standardized_filenames():
            filepath = Path(dest, name)
            rec = rec.to_symlink(filepath, overwrite=True)
            result.add(rec)
        return result


# This is how we originally handled longitudinal references, but the approach worked poorly.
# Now longitudinal data is caputed using PatientID/StudyDate/Study/... file structure.
# This class remains for reproducibility and will be removed in a future version
class LongitudinalPointerOutput(Output):
    def __call__(self, inp: Union[Input, Dict[str, RecordCollection]]) -> Dict[str, RecordCollection]:
        assert isinstance(inp, dict)
        # build lookup for years and related cases
        year_lookup = self.create_year_lookup(inp)
        association_lookup = self.create_association_lookup(inp)
        reverse_association_lookup = {name: group for group, names in association_lookup.items() for name in names}

        result: Dict[str, RecordCollection] = {}
        bar = self.tqdm(inp if isinstance(inp, Input) else inp.items())
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


class FileListOutput(Output):
    def __init__(
        self,
        path: PathLike,
        filename: PathLike,
        record_filter: Optional[Callable[[FileRecord], bool]] = None,
        collection_filter: Optional[Callable[[RecordCollection], bool]] = None,
        use_bar: bool = True,
        by_case: Optional[bool] = None,
    ):
        super().__init__(path, record_filter, collection_filter, use_bar)
        self.filename = Path(filename)
        self.by_case = by_case if by_case is not None else (self.record_filter is None)

    def __call__(self, inp: Union[Input, Dict[str, RecordCollection]]) -> Dict[str, RecordCollection]:
        assert isinstance(inp, dict)
        result: Dict[str, RecordCollection] = {}
        bar = self.tqdm(inp if isinstance(inp, Input) else inp.items())
        dest = Path(self.path, self.filename)

        f = open(dest, "w")
        for name, collection in bar:
            # skip entire collection if collection_filter is True
            if self.collection_filter is not None and not self.collection_filter(collection):
                continue

            # filter records
            if self.record_filter:
                collection = collection.filter(self.record_filter)
            if all_records_were_filtered := not collection:
                continue

            if self.by_case:
                f.write(f"{name}\n")
            else:
                files = [self.path_to_filelist_entry(rec.path) for rec in collection]
                for p in files:
                    f.write(f"{str(p)}\n")

            result[name] = collection
        f.close()

        # remove empty file
        if not dest.stat().st_size:
            dest.unlink()

        return result

    @classmethod
    def path_to_filelist_entry(cls, path: Path) -> Path:
        # For file structure PatientID/StudyDate/Study/file we want
        # to select 4 levels from the end to trim leading directories
        START_OF_PATH = -4
        return Path(*path.parts[START_OF_PATH:])

    def write(self, name: str) -> RecordCollection:
        return RecordCollection()


def is_complete_case(c: RecordCollection) -> bool:
    mammograms = c.filter(lambda rec: isinstance(rec, MammogramFileRecord))
    assert all(isinstance(rec, MammogramFileRecord) for rec in mammograms)
    mammograms = cast(Iterable[MammogramFileRecord], mammograms)
    return MammogramFileRecord.is_complete_mammo_case(mammograms)


def is_mammogram_case(c: RecordCollection) -> bool:
    return any(isinstance(rec, MammogramFileRecord) for rec in c)


def is_mammogram_record(
    rec: FileRecord, mtype: Optional[MammogramType] = None, secondary: bool = False, proc: bool = False
) -> bool:
    return (
        isinstance(rec, MammogramFileRecord)
        and (mtype is None or rec.mammogram_type == mtype)
        and (secondary or not rec.is_secondary_capture)
        and (proc or not rec.is_for_processing)
    )


def is_2d_mammogram(rec: FileRecord) -> bool:
    return isinstance(rec, MammogramFileRecord) and rec.is_2d


def is_spot_mag(rec: FileRecord) -> bool:
    return isinstance(rec, MammogramFileRecord) and (rec.is_spot_compression or rec.is_magnified)


def is_standard_ffdm(
    rec: FileRecord,
    secondary: bool = False,
    proc: bool = False,
) -> bool:
    return (
        isinstance(rec, MammogramFileRecord)
        and (rec.mammogram_type == MammogramType.FFDM)
        and (rec.is_standard_mammo_view)
        and (secondary or not rec.is_secondary_capture)
        and (proc or not rec.is_for_processing)
    )


# register primary output groups
# these should not use a record filter
PRIMARY_OUTPUT_GROUPS = [
    ("cases", None),
    ("mammograms", is_mammogram_case),
    ("complete-mammograms", is_complete_case),
]
for name, collection_filter in PRIMARY_OUTPUT_GROUPS:
    OUTPUT_REGISTRY(
        partial(SymlinkFileOutput, collection_filter=collection_filter),
        name=f"symlink-{name}",
        subdir=name,
    )
    OUTPUT_REGISTRY(
        partial(
            FileListOutput,
            filename=Path(f"{name}.txt"),
            collection_filter=collection_filter,
        ),
        name=f"filelist-{name}",
        subdir="file_lists/by_case",
        derived=True,
    )

# register primary output groups
# these can use a record filter
SECONDARY_OUTPUT_GROUPS: List[Tuple[str, Callable, Callable]] = [
    (mtype.simple_name, is_mammogram_case, partial(is_mammogram_record, mtype=mtype))
    for mtype in MammogramType
    if mtype != MammogramType.UNKNOWN
]
SECONDARY_OUTPUT_GROUPS.append(("spot_mag", is_mammogram_case, is_spot_mag))
for name, collection_filter, record_filter in SECONDARY_OUTPUT_GROUPS:
    for by_case in (False, True):
        by_case_str = "by_case" if by_case else "by_file"
        OUTPUT_REGISTRY(
            partial(
                FileListOutput,
                filename=Path(f"{name}.txt"),
                collection_filter=collection_filter,
                record_filter=record_filter,
                by_case=by_case,
            ),
            name=f"filelist-{name}-{by_case_str}",
            subdir=f"file_lists/{by_case_str}",
            derived=True,
        )

for by_case in (False, True):
    by_case_str = "by_case" if by_case else "by_file"
    OUTPUT_REGISTRY(
        partial(
            FileListOutput,
            filename=Path("ffdm_complete.txt"),
            collection_filter=is_complete_case,
            record_filter=is_standard_ffdm,
            by_case=by_case,
        ),
        name=f"filelist-standard_ffdm-{by_case_str}",
        subdir=f"file_lists/{by_case_str}",
        derived=True,
    )

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from registry import Registry
from tqdm_multiprocessing import ConcurrentMapper

from ..types import MammogramType
from .collection import CollectionHelper, RecordCollection
from .collection import apply_helpers as apply_collection_helpers
from .collection import get_bar_description_suffix
from .input import Input
from .protocols import SupportsGenerated
from .record import HELPER_REGISTRY, FileRecord, MammogramFileRecord, RecordHelper, apply_helpers


YEAR_RE: Final = re.compile(r"\d{4}")

OUTPUT_REGISTRY = Registry("output", bound=Callable[..., "Output"])

R = TypeVar("R", bound=FileRecord)

# For file structure cases/PatientID/Study/file we want
# to select 4 levels from the end to trim leading directories
FILELIST_OFFSET: Final[int] = 4

# For file structure cases/PatientID/Study we want
# to select 3 levels from the end to trim leading directories
CASELIST_OFFSET: Final[int] = 3


@dataclass
class WriteResult:
    key: Tuple[str, ...]
    path: Path
    collection: RecordCollection
    name: str
    metadata: Any = field(repr=False, default=None)

    def __hash__(self) -> int:
        return hash(self.key)


def iterate_symlinks(collection: RecordCollection[R], dest: Path) -> Iterator[Tuple[Path, R]]:
    for pathname, rec in collection.standardized_filenames():
        filepath = Path(dest, pathname)
        yield filepath, rec


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

        helpers:
            List of helper names to apply to the output. Can be :class:`RecordHelper` or
            :class:`CollectionHelper` instances.

        use_bar:
            Whether to use a progress bar.

        threads:
            Whether to use threads instead of processes.

        jobs:
            Number of jobs to use. If None, use the number of CPUs.

        chunksize:
            Passed to :class:`ConcurrentMapper`.
    """

    def __init__(
        self,
        root: PathLike,
        output_subdir: PathLike = Path("."),
        record_filter: Optional[Callable[[FileRecord], bool]] = None,
        collection_filter: Optional[Callable[[RecordCollection], bool]] = None,
        helpers: List[str] = [],
        use_bar: bool = True,
        threads: bool = False,
        jobs: Optional[int] = None,
        chunksize: int = 1,
        timeout: Optional[int] = None,
    ):
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(root)
        self.output_subdir = Path(output_subdir)
        self.path.mkdir(exist_ok=True, parents=True)
        self.record_filter = record_filter
        self.collection_filter = collection_filter
        self.use_bar = use_bar
        self.threads = threads
        self.jobs = jobs
        self.chunksize = chunksize
        self.timeout = timeout
        helper_fns = [HELPER_REGISTRY.get(h).instantiate_with_metadata().fn for h in helpers]
        self.record_helpers: List[RecordHelper] = [h for h in helper_fns if isinstance(h, RecordHelper)]
        self.collection_helpers: List[CollectionHelper] = [h for h in helper_fns if isinstance(h, CollectionHelper)]

    @property
    def path(self) -> Path:
        return self.root / self.output_subdir

    def mapper(self, **kwargs) -> ConcurrentMapper:
        mapper = ConcurrentMapper(self.threads, self.jobs, chunksize=self.chunksize, timeout=self.timeout)
        kwargs.setdefault("disable", not self.use_bar)
        kwargs.setdefault("leave", False)
        mapper.create_bar(**kwargs)
        return mapper

    def __call__(self, inp: Union[Input, Dict[str, Iterable[WriteResult]]]) -> Dict[str, Iterable[WriteResult]]:
        result: Dict[str, Iterable[WriteResult]] = {}
        iterable = inp if isinstance(inp, Input) else inp.items()
        desc = get_bar_description_suffix(self.jobs, self.threads)
        with self.mapper(total=len(iterable), desc=f"Writing {self.__class__.__name__} ({desc})") as mapper:
            it = mapper(
                self._process,
                iterable,
                path=self.path,
                record_filter=self.record_filter,
                collection_filter=self.collection_filter,
            )
            for item in it:
                if item:
                    name = next(x.name for x in item)
                    result[name] = item
        return result

    def _process(
        self,
        kc: Tuple[Sequence[str], RecordCollection],
        path: Path,
        record_filter: Optional[Callable[[FileRecord], bool]] = None,
        collection_filter: Optional[Callable[[RecordCollection], bool]] = None,
    ) -> Optional[Iterable[WriteResult]]:
        key, collection = kc
        name = str(Path(*key))
        # apply record and collection helpers
        if self.record_helpers:
            collection = RecordCollection((apply_helpers(rec.path, rec, self.record_helpers) for rec in collection))
        if self.collection_helpers:
            index = len(key)
            # since we will often use multiple outputs, ensure that the original collection cannot be mutated by
            # passing a copy below
            collection = apply_collection_helpers(copy(collection), self.collection_helpers, index)
            assert isinstance(
                collection, RecordCollection
            ), f"Collection helpers must return a RecordCollection, got {type(collection)}"

        # apply collection and record filters
        if collection_filter is not None and not collection_filter(collection):
            return
        if record_filter is not None:
            collection = collection.filter(record_filter)
            assert isinstance(
                collection, RecordCollection
            ), f"Record filters must return a RecordCollection, got {type(collection)}"

        dest = Path(path, name)
        # pyright fails to recognize this is an abstract classmethod
        written = self.write(key, name, collection, dest)  # type: ignore
        return list(written)

    @abstractmethod
    def write(
        self, key: Tuple[str, ...], name: str, collection: RecordCollection, dest: Path
    ) -> Iterator[WriteResult]: ...


class SymlinkFileOutput(Output):
    def write(self, key: Tuple[str, ...], name: str, collection: RecordCollection, dest: Path) -> Iterator[WriteResult]:
        dest.mkdir(exist_ok=True, parents=True)
        result: RecordCollection[FileRecord] = RecordCollection()
        for filepath, rec in iterate_symlinks(collection, dest):
            assert isinstance(rec, FileRecord)
            generated = isinstance(rec, SupportsGenerated) and rec.generated
            if rec.exists and not generated:
                rec = rec.to_symlink(filepath, overwrite=True)
            result.add(rec)
            yield WriteResult(key, filepath, result, name, None)


class ManifestOutput(Output):
    def __call__(self, inp: Union[Input, Dict[str, Iterable[WriteResult]]]) -> Dict[str, Iterable[WriteResult]]:
        written = super().__call__(inp)
        manifest = self.build_main_manifest(written)
        path = Path(self.root, "manifest.json")
        with open(path, "w") as f:
            json.dump(manifest, f, sort_keys=True, default=str)
        return written

    @classmethod
    def build_main_manifest(cls, values: Dict[str, Iterable[WriteResult]]) -> Dict[str, Any]:
        result = dict()
        for name, write_results in values.items():
            for write_result in write_results:
                container = result
                for k in write_result.key:
                    container = container.setdefault(k, {})
                if write_result.metadata:
                    container.update(write_result.metadata)
        return result

    def write(
        self,
        key: Tuple[str, ...],
        name: str,
        collection: RecordCollection,
        dest: Path,
    ) -> Iterator[WriteResult]:
        # we want the manifest to reflect symlink paths and real paths.
        symlink_collection = RecordCollection()
        for symlink_path, rec in iterate_symlinks(collection, dest):
            symlink_rec = rec.replace(path=Path(symlink_path.absolute()))
            symlink_collection.add(symlink_rec)
        collection = symlink_collection

        metadata = collection.to_dict()
        metadata["name"] = name

        dest.mkdir(exist_ok=True, parents=True)
        path = Path(dest, "manifest.json")
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True, default=str)
        yield WriteResult(key, path, collection, name=name, metadata=metadata)


# TODO: consider deprecating this
class FileListOutput(SymlinkFileOutput):
    def __init__(
        self,
        root: PathLike,
        output_subdir: PathLike = Path("."),
        filelist_output_subdir: PathLike = Path("filelist"),
        record_filter: Optional[Callable[[FileRecord], bool]] = None,
        collection_filter: Optional[Callable[[RecordCollection], bool]] = None,
        helpers: List[str] = [],
        use_bar: bool = True,
        threads: bool = False,
        jobs: Optional[int] = None,
        chunksize: int = 8,
        by_case: Optional[bool] = None,
        min_name_len: Optional[int] = 3,
    ):
        super().__init__(
            root, output_subdir, record_filter, collection_filter, helpers, use_bar, threads, jobs, chunksize
        )
        self.filelist_output_subdir = Path(filelist_output_subdir)
        self.by_case = by_case if by_case is not None else (self.record_filter is None)
        self.min_name_len = min_name_len

    @property
    def filelist_path(self) -> Path:
        return self.root / self.filelist_output_subdir

    def write(self, key: Tuple[str, ...], name: str, collection: RecordCollection, dest: Path) -> Iterator[WriteResult]:
        iterator = (
            self._iterate_case_entries(collection, dest)
            if self.by_case
            else self._iterate_file_entries(collection, dest)
        )
        for entry in iterator:
            yield WriteResult(key, dest, collection, name, metadata=entry)

    def _iterate_file_entries(self, collection: RecordCollection, dest: Path) -> Iterator[str]:
        for filepath, rec in iterate_symlinks(collection, dest):
            yield str(self.path_to_filelist_entry(filepath))

    def _iterate_case_entries(self, collection: RecordCollection, dest: Path) -> Iterator[str]:
        seen_entries: Set[Path] = set()
        for file_entry in self._iterate_file_entries(collection, dest):
            entry = Path(file_entry).parent
            if entry in seen_entries:
                continue
            if not self.min_name_len or len(entry.parts) >= self.min_name_len:
                seen_entries.add(entry)
                yield str(entry)

    def __call__(self, inp: Union[Input, Dict[str, Iterable[WriteResult]]]) -> Dict[str, Iterable[WriteResult]]:
        written = super().__call__(inp)
        filelist = self.build_filelist(written)
        dest = Path(self.filelist_path)
        dest.parent.mkdir(exist_ok=True, parents=True)
        with open(dest, "w") as f:
            for name, entries in filelist.items():
                for entry in entries:
                    f.write(f"{entry}\n")
        return written

    def build_filelist(self, values: Dict[str, Iterable[WriteResult]]) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {}
        for name, write_results in values.items():
            for write_result in write_results:
                assert isinstance(write_result.metadata, str)
                container = result.setdefault(name, [])
                container.append(write_result.metadata)
        return result

    def path_to_filelist_entry(self, path: Path) -> Path:
        return path.relative_to(self.root)


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
        SymlinkFileOutput,
        name=f"symlink-{name}",
        output_subdir=name,
        collection_filter=collection_filter,
    )
    OUTPUT_REGISTRY(
        FileListOutput,
        name=f"filelist-{name}",
        output_subdir=name,
        filelist_output_subdir=Path(f"file_lists/by_case/{name}.txt"),
        collection_filter=collection_filter,
    )

OUTPUT_REGISTRY(ManifestOutput, name="manifest", output_subdir="cases")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from argparse import ArgumentParser
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pydicom
from tqdm import tqdm

from ..dicom import has_dicm_prefix


@dataclass
class FileCheckResult:
    series_uid: str
    study_uid: str
    path: Path

    def __hash__(self) -> int:
        return hash(self.series_uid + self.study_uid + str(self.path))

    def check_study_overlap(self, other: "FileCheckResult") -> bool:
        if not (self.series_uid and other.series_uid):
            return False

        if self.study_uid == other.study_uid:
            if self.path.parent != other.path.parent:
                return True
        return False

    def check_series_overlap(self, other: "FileCheckResult") -> bool:
        if not (self.series_uid and other.series_uid):
            return False

        if self.series_uid == other.series_uid:
            if self.path != other.path:
                return True
        return False


def get_parser(parser: ArgumentParser = ArgumentParser()) -> ArgumentParser:
    parser.add_argument("path", help="path to search for duplicates")
    parser.add_argument("check", choices=["series", "study"], help="whether to check series or study overlap")
    parser.add_argument("--name", "-n", default="*", help="glob pattern for filename")
    parser.add_argument("--jobs", "-j", default=4, help="parallelism")
    return parser


def read_file(path: Path, args: argparse.Namespace) -> Optional[FileCheckResult]:
    if not path.is_file() or not has_dicm_prefix(path):
        return

    try:
        tags = ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"]
        with pydicom.dcmread(path, specific_tags=tags, stop_before_pixels=True) as dcm:
            study = getattr(dcm, "StudyInstanceUID", "")
            series = getattr(dcm, "SeriesInstanceUID", "")
            sop = getattr(dcm, "SOPInstanceUID", "")
    except AttributeError:
        return

    return FileCheckResult(sop if sop else series, study, path)


def should_print(printed: Set[Tuple[Path, Path]], p1: Path, p2: Path) -> bool:
    return not ((p1, p2) in printed or (p2, p1) in printed)


def check_overlap(
    seen: Dict[str, Set[FileCheckResult]],
    series: bool = True,
) -> Set[Tuple[Path, Path]]:
    results: Set[Tuple[Path, Path]] = set()
    for result_set in seen.values():
        for r1, r2 in product(result_set, result_set):
            if series and r1.check_series_overlap(r2):
                p1, p2 = r1.path, r2.path
            elif not series and r1.check_study_overlap(r2):
                p1, p2 = r1.path.parent, r2.path.parent
            else:
                continue

            if should_print(results, p1, p2):
                results.add((p1, p2))
                tqdm.write(f"{p1}\t{p2}")

    return results


def main(args: argparse.Namespace) -> None:
    path = Path(args.path)
    if not path.is_dir():
        raise NotADirectoryError(path)

    seen_series: Dict[str, Set[FileCheckResult]] = {}
    seen_studies: Dict[str, Set[FileCheckResult]] = {}
    bar = tqdm(desc="Finding files")

    def callback(x: Future):
        bar.update(1)
        result: FileCheckResult = x.result()
        if result is None:
            return

        _seen_series = seen_series.get(result.series_uid, set())
        _seen_series.add(result)
        _seen_studies = seen_studies.get(result.study_uid, set())
        _seen_studies.add(result)
        seen_series[result.series_uid] = _seen_series
        seen_studies[result.study_uid] = _seen_studies

    futures: List[Future] = []
    with ThreadPoolExecutor(args.jobs) as tp:
        for match in path.rglob(args.name):
            f = tp.submit(read_file, match, args)
            f.add_done_callback(callback)
            futures.append(f)
    bar.close()

    series = True if args.check == "series" else False
    check_overlap(seen_series, series=series)


def entrypoint():
    parser = get_parser()
    main(parser.parse_args())


if __name__ == "__main__":
    entrypoint()

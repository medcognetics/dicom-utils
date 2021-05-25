#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from argparse import ArgumentParser
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pydicom

from ..dicom import has_dicm_prefix


def get_parser(parser: ArgumentParser = ArgumentParser()) -> ArgumentParser:
    parser.add_argument("path1", help="first path to search")
    parser.add_argument("path2", help="second path to search")
    parser.add_argument("--name", "-n", default="*", help="glob pattern for filename")
    parser.add_argument("--parents", "-p", default=True, action="store_true", help="return unique parent directories")
    parser.add_argument("--jobs", "-j", default=4, help="parallelism")
    # TODO add a field to only return DICOMs with readable image data
    return parser


def check_file(path: Path, args: argparse.Namespace) -> Optional[Tuple[Path, Optional[str]]]:
    if not path.is_file() or not has_dicm_prefix(path):
        return

    try:
        tags = ["StudyInstanceUID"]
        with pydicom.dcmread(path, specific_tags=tags, stop_before_pixels=True) as dcm:
            study = getattr(dcm, "StudyInstanceUID", None)
    except AttributeError:
        return

    if study is None:
        return

    if args.parents:
        path = path.parent

    return path, study


def main(args: argparse.Namespace) -> None:
    path1 = Path(args.path1)
    path2 = Path(args.path2)
    if not path1.is_dir():
        raise NotADirectoryError(path1)
    if not path2.is_dir():
        raise NotADirectoryError(path2)

    path1_seen: Dict[str, Set[Path]] = {}
    path2_seen: Dict[str, Set[Path]] = {}

    def callback(x: Future):
        result = x.result()
        if result is None:
            return
        path, study = result

        if path.is_relative_to(args.path1):
            container = path1_seen
        elif path.is_relative_to(args.path2):
            container = path2_seen
        else:
            raise RuntimeError()

        seen = container.get(study, set())
        seen.add(path)
        container[study] = seen

    futures: List[Future] = []
    with ThreadPoolExecutor(args.jobs) as tp:
        for match in path1.rglob(args.name):
            f = tp.submit(check_file, match, args)
            f.add_done_callback(callback)
            futures.append(f)
        for match in path2.rglob(args.name):
            f = tp.submit(check_file, match, args)
            f.add_done_callback(callback)
            futures.append(f)

    for study, pathset1 in path1_seen.items():
        pathset2 = path2_seen.get(study, {})

        if not pathset2:
            continue

        for p in pathset1:
            print(f"{study}\t{p}")
        for p in pathset2:
            print(f"{study}\t{p}")


def entrypoint():
    parser = get_parser()
    main(parser.parse_args())


if __name__ == "__main__":
    entrypoint()

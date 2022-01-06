#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from argparse import ArgumentParser
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Set

import statistics
import pydicom

from ..dicom import has_dicm_prefix, num_pixels
from ..types import SimpleImageType


def get_parser(parser: ArgumentParser = ArgumentParser()) -> ArgumentParser:
    parser.add_argument("path", default="./", help="path to find from")
    parser.add_argument("--name", "-n", default="*", help="glob pattern for filename")
    parser.add_argument("--parents", "-p", default=False, action="store_true", help="return unique parent directories")
    parser.add_argument(
        "--types", "-t", choices=["2d", "tomo", "s-view"], default=None, nargs="+", help="filter by image type"
    )
    parser.add_argument("--jobs", "-j", default=4, help="parallelism")
    parser.add_argument("--core-only", default=False, action="store_true", help="try to ignore")
    parser.add_argument("--extra-only", default=False, action="store_true", help="try to ignore")
    # TODO add a field to only return DICOMs with readable image data
    return parser


def is_desired_type(path: Path, types: List[str]) -> bool:
    with pydicom.dcmread(path, stop_before_pixels=True) as dcm:
        simple_image_type = SimpleImageType.from_dicom(dcm)
    return str(simple_image_type) in types


def check_file(path: Path, args: argparse.Namespace) -> Optional[Path]:
    if not path.is_file() or not has_dicm_prefix(path):
        return
    try:
        if args.types is not None and not is_desired_type(path, args.types):
            return
    except Exception:
        return

    return path


def filter_core(path: Path) -> Set[Path]:
    all_dicoms = {
        file: num_pixels(file)
        for file in path.glob("*")
        if file.is_file() and has_dicm_prefix(file)
    }
    core_size = max(statistics.multimode(all_dicoms.values()))
    return {file for file, size in all_dicoms.items() if size == core_size}


def main(args: argparse.Namespace) -> None:
    path = Path(args.path)
    if not path.is_dir():
        raise NotADirectoryError(path)

    seen_parents = set()

    def callback(x: Future):
        result = x.result()
        if result is None:
            return

        try:
            if args.parents and result.parent not in seen_parents:
                print(result.parent, flush=True)
            elif not args.parents:
                if args.core_only and result in filter_core(result.parent):
                    print(result, flush=True)
                elif args.extra_only and result not in filter_core(result.parent):
                    print(result, flush=True)
                elif not (args.extra_only or args.core_only):
                    print(result, flush=True)
            seen_parents.add(result.parent)
        except IOError:
            tp.shutdown(wait=False)

    futures: List[Future] = []
    with ThreadPoolExecutor(args.jobs) as tp:
        for match in path.rglob(args.name):
            f = tp.submit(check_file, match, args)
            f.add_done_callback(callback)
            futures.append(f)


def entrypoint():
    parser = get_parser()
    main(parser.parse_args())


if __name__ == "__main__":
    entrypoint()

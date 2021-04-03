#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import pydicom

from .dicom import has_dicm_prefix
from .metadata import get_simple_image_type, process_image_type


def get_parser(parser: ArgumentParser = ArgumentParser()) -> ArgumentParser:
    parser.add_argument("path", default="./", help="path to find from")
    parser.add_argument("--name", "-n", default="*", help="glob pattern for filename")
    parser.add_argument("--parents", "-p", default=False, action="store_true", help="return unique parent directories")
    parser.add_argument(
        "--types", "-t", choices=["2d", "tomo", "s-view"], default=None, nargs="+", help="filter by image type"
    )
    # TODO add a field to only return DICOMs with readable image data
    return parser


def is_desired_type(path: Path, types: List[str]) -> bool:
    with pydicom.dcmread(path, stop_before_pixels=True) as dcm:
        image_type = process_image_type(dcm)
    simple_image_type = get_simple_image_type(image_type)
    return str(simple_image_type) in types


def main(args: argparse.Namespace) -> None:
    path = Path(args.path)
    if not path.is_dir():
        raise NotADirectoryError(path)

    seen_parents = set()
    for match in path.rglob(args.name):
        if not match.is_file() or not has_dicm_prefix(match):
            continue
        try:
            if args.types is not None and not is_desired_type(match, args.types):
                continue
        except Exception:
            continue

        if args.parents and match.parent not in seen_parents:
            print(match.parent)
            seen_parents.add(match.parent)
        elif not args.parents:
            print(match)

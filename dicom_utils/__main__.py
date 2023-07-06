#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import inspect
from argparse import ArgumentParser, _SubParsersAction
from typing import Callable

from .cli.cat import get_parser as cat_parser
from .cli.cat import main as cat_main
from .cli.decompress import get_parser as decompress_parser
from .cli.decompress import main as decompress_main
from .cli.dicom2img import get_parser as dicom2img_parser
from .cli.dicom2img import main as dicom2img_main
from .cli.dicom_types import get_parser as dicom_types_parser
from .cli.dicom_types import main as dicom_types_main
from .cli.dicomphi import get_parser as dicomphi_parser
from .cli.dicomphi import main as dicomphi_main
from .cli.find import get_parser as find_parser
from .cli.find import main as find_main
from .cli.merge_series import get_parser as merge_series_parser
from .cli.merge_series import main as merge_series_main
from .cli.overlap import get_parser as overlap_parser
from .cli.overlap import main as overlap_main
from .cli.strip import get_parser as strip_parser
from .cli.strip import main as strip_main
from .cli.validate import get_parser as validate_parser
from .cli.validate import main as validate_main
from .logging import LoggingLevel, set_logging_level


Main = Callable[[argparse.Namespace], None]
Modifier = Callable[[ArgumentParser], None]


def assert_name_match(name: str, main: Main, modifier: Modifier) -> None:
    # Enforce a naming convention to reduce chance of bugs
    main_module = inspect.getmodule(main)
    assert main_module
    assert name in (module := main_module.__name__), f"'{name}' not in '{module}'"
    modifier_module = inspect.getmodule(modifier)
    assert modifier_module
    assert name in (module := modifier_module.__name__), f"'{name}' not in '{module}'"


def add_subparser(subparsers: _SubParsersAction, name: str, help: str, main: Main, modifier: Modifier) -> None:
    assert_name_match(name, main, modifier)
    subparser = subparsers.add_parser(name, help=help)
    subparser.set_defaults(main=main)
    modifier(subparser)
    ll = LoggingLevel.list()
    subparser.add_argument("--logging_level", "-ll", help="set logging level", choices=ll, default=LoggingLevel.WARNING)
    subparser.add_argument("--pydicom_logging_level", "-pl", help="set pydicom logging level", choices=ll, default=None)


def main() -> None:
    parser = ArgumentParser(description="DICOM CLI utilities")

    subparsers = parser.add_subparsers(help="Operation modes")
    for name, help, main, modifier in [
        ("cat", "Print DICOM metadata", cat_main, cat_parser),
        ("dicom2img", "Convert DICOM to image file", dicom2img_main, dicom2img_parser),
        ("dicomphi", "Find PHI in DICOMs", dicomphi_main, dicomphi_parser),
        ("find", "Find DICOM files", find_main, find_parser),
        ("strip", "Strip pixel data out of DICOMs", strip_main, strip_parser),
        ("dicom_types", "Summarize image types", dicom_types_main, dicom_types_parser),
        ("overlap", "Check overlap of study UIDs between dirs", overlap_main, overlap_parser),
        ("validate", "Validate metadata of a DICOM file", validate_main, validate_parser),
        ("decompress", "Decompress pixel contents of a DICOM file", decompress_main, decompress_parser),
        ("merge_series", "Merge series DICOMs into a single DICOM", merge_series_main, merge_series_parser),
    ]:
        add_subparser(subparsers, name=name, help=help, main=main, modifier=modifier)

    args = parser.parse_args()

    if hasattr(args, "main"):
        set_logging_level(args.logging_level, args.pydicom_logging_level)
        args.main(args)
    else:
        parser.print_usage()


if __name__ == "__main__":
    main()

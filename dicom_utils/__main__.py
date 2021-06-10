#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from argparse import ArgumentParser, _SubParsersAction
from typing import Callable

from .cli.cat import get_parser as cat_parser
from .cli.cat import main as cat_main
from .cli.dicom2img import get_parser as dicom2img_parser
from .cli.dicom2img import main as dicom2img_main
from .cli.dicom_types import get_parser as dicom_types_parser
from .cli.dicom_types import main as dicom_types_main
from .cli.find import get_parser as find_parser
from .cli.find import main as find_main
from .cli.overlap import get_parser as overlap_parser
from .cli.overlap import main as overlap_main
from .logging import LoggingLevel, set_logging_level


Main = Callable[[argparse.Namespace], None]
Modifier = Callable[[ArgumentParser], None]


def add_subparser(subparsers: _SubParsersAction, name: str, help: str, main: Main, modifier: Modifier) -> None:
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
        ("find", "Find DICOM files", find_main, find_parser),
        ("dicom_types", "Summarize image types", dicom_types_main, dicom_types_parser),
        ("overlap", "Check overlap of study UIDs between dirs", overlap_main, overlap_parser),
    ]:
        add_subparser(subparsers, name=name, help=help, main=main, modifier=modifier)

    args = parser.parse_args()
    set_logging_level(args.logging_level, args.pydicom_logging_level)

    if hasattr(args, "main"):
        args.main(args)
    else:
        parser.print_usage()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser

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


def main() -> None:
    parser = ArgumentParser(description="DICOM CLI utilities")
    subparsers = parser.add_subparsers(help="Operation modes")

    subparser = subparsers.add_parser("cat", help="Print DICOM metadata")
    subparser.set_defaults(func=cat_main)
    cat_parser(subparser)

    subparser = subparsers.add_parser("dicom2img", help="Convert DICOM to image file")
    subparser.set_defaults(func=dicom2img_main)
    dicom2img_parser(subparser)

    subparser = subparsers.add_parser("find", help="Find DICOM files")
    subparser.set_defaults(func=find_main)
    find_parser(subparser)

    subparser = subparsers.add_parser("dicom_types", help="Summarize image types")
    subparser.set_defaults(func=dicom_types_main)
    dicom_types_parser(subparser)

    subparser = subparsers.add_parser("overlap", help="Check overlap of study UIDs between dirs")
    subparser.set_defaults(func=overlap_main)
    overlap_parser(subparser)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_usage()


if __name__ == "__main__":
    main()

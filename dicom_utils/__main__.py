#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser

from .cat import get_parser as cat_parser
from .cat import main as cat_main
from .dicom2img import get_parser as dicom2img_parser
from .dicom2img import main as dicom2img_main
from .find import get_parser as find_parser
from .find import main as find_main


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

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_usage()


if __name__ == "__main__":
    main()

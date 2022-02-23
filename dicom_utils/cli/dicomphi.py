#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from argparse import ArgumentParser

from ..find_phi import find_phi


def get_parser(parser: ArgumentParser = ArgumentParser()) -> ArgumentParser:
    parser.add_argument("path", help="path to a DICOM file or folder with DICOM files")
    parser.add_argument(
        "--overwrite", help="overwrite original files with anonymized files", default=False, action="store_true"
    )
    parser.add_argument("--verbose", "-v", help="verbose printing", default=False, action="store_true")
    return parser


def main(args: argparse.Namespace) -> None:
    print(f"Overwrite is set to {args.overwrite}")
    print(f"Verbose is set to {args.verbose}")
    find_phi(args.path, args.overwrite, args.verbose)


def entrypoint() -> None:
    main(get_parser().parse_args())


if __name__ == "__main__":
    entrypoint()

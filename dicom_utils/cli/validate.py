#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import sys
from argparse import ArgumentParser
from pathlib import Path

from ..validation import Validator


def get_parser(parser: ArgumentParser = ArgumentParser()) -> ArgumentParser:
    parser.add_argument("path", help="DICOM file to validate")
    parser.add_argument("-f", "--failing-only", default=False, action="store_true", help="only print failing tags")
    parser.add_argument("-n", "--no-color", default=False, action="store_true", help="disable printing with color")
    return parser


def main(args: argparse.Namespace) -> bool:
    src = Path(args.path)
    if not src.is_file():
        raise FileNotFoundError(src)

    validator = Validator.default_validator()
    results = validator.validate_dicom_file(src)
    validator.report_string(results, args.failing_only, not args.no_color)
    return validator.all_passing(results)


def entrypoint():
    result = main(get_parser().parse_args())
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    entrypoint()

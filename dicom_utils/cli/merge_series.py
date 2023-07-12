#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from argparse import ArgumentParser
from pathlib import Path

from ..merge_series import merge_series


def get_parser(parser: ArgumentParser = ArgumentParser()) -> ArgumentParser:
    parser.add_argument("path", help="path to a folder with DICOM files", type=Path)
    return parser


def main(args: argparse.Namespace) -> None:
    merge_series(Path(args.path))


def entrypoint() -> None:
    main(get_parser().parse_args())


if __name__ == "__main__":
    entrypoint()

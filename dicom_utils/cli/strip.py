#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from argparse import ArgumentParser
from pathlib import Path
from typing import Final

import pydicom
from pydicom import Dataset


def get_parser(parser: ArgumentParser = ArgumentParser()) -> ArgumentParser:
    parser.add_argument("path", help="DICOM file with pixel data to remove")
    parser.add_argument("-o", "--output", help="path for stripped file else original path is used", default=None)
    return parser


def strip_pixel_data(ds: Dataset) -> None:
    del ds.PixelData


def to_stripped_dicom(src: Path, dst: Path) -> None:
    ds = pydicom.dcmread(src)
    strip_pixel_data(ds)
    ds.save_as(dst)


def main(args: argparse.Namespace) -> None:
    src_path: Final[Path] = Path(args.path)
    dst_path: Final[Path] = Path(args.output or src_path)
    to_stripped_dicom(src_path, dst_path)


def entrypoint():
    main(get_parser().parse_args())


if __name__ == "__main__":
    entrypoint()

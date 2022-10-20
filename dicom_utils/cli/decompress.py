#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from argparse import ArgumentParser
from pathlib import Path

import pydicom
from pydicom import FileDataset
from pydicom.uid import ExplicitVRLittleEndian

from ..dicom import set_pixels
from ..types import Dicom


def get_parser(parser: ArgumentParser = ArgumentParser()) -> ArgumentParser:
    parser.add_argument("path", type=Path, help="DICOM file to decompress")
    parser.add_argument("dest", type=Path, help="destination DICOM filepath")
    parser.add_argument(
        "-s", "--strict", default=False, action="store_true", help="if true, raise an error on uncompressed input"
    )
    return parser


def decompress(dcm: Dicom, strict: bool = False) -> Dicom:
    tsuid = dcm.file_meta.TransferSyntaxUID
    if not tsuid.is_compressed:
        if strict:
            raise RuntimeError(f"TransferSyntaxUID {tsuid} is already decompressed")
        else:
            return dcm

    pixels = dcm.pixel_array
    assert isinstance(dcm, FileDataset)
    dcm = set_pixels(dcm, pixels, ExplicitVRLittleEndian)
    return dcm


def main(args: argparse.Namespace) -> None:
    path = Path(args.path)
    dest = Path(args.dest)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not dest.parent.is_dir():
        raise NotADirectoryError(dest.parent)

    with pydicom.dcmread(path) as dcm:
        dcm = decompress(dcm, strict=args.strict)
        assert not dcm.file_meta.TransferSyntaxUID.is_compressed
        dcm.save_as(dest)


def entrypoint():
    parser = get_parser()
    main(parser.parse_args())


if __name__ == "__main__":
    entrypoint()

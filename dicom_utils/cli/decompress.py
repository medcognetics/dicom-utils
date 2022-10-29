#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from argparse import ArgumentParser
from pathlib import Path
from time import time

import numpy as np
import pydicom
from PIL import Image

from ..dicom import decompress


def get_parser(parser: ArgumentParser = ArgumentParser()) -> ArgumentParser:
    parser.add_argument("path", type=Path, help="DICOM file to decompress")
    parser.add_argument("dest", type=Path, help="destination DICOM filepath")
    parser.add_argument(
        "-s", "--strict", default=False, action="store_true", help="if true, raise an error on uncompressed input"
    )
    parser.add_argument(
        "-g", "--gpu", default=False, action="store_true", help="use NVJPEG2K accelerated decompression"
    )
    parser.add_argument(
        "-b", "--batch-size", default=4, type=int, help="batch size for NVJPEG2K accelerated decompression"
    )
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="print NVJPEG2K outputs")
    parser.add_argument(
        "-t", "--test", default=False, action="store_true", help="test equivalence to CPU decompression"
    )
    return parser


def to_img(x: np.ndarray) -> Image.Image:
    x = x.astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min()) * 255
    x = x.astype(np.uint8)
    return Image.fromarray(x)


def main(args: argparse.Namespace) -> None:
    path = Path(args.path)
    dest = Path(args.dest)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not dest.parent.is_dir():
        raise NotADirectoryError(dest.parent)

    start_time = time()
    with pydicom.dcmread(path) as dcm:
        dcm = decompress(dcm, strict=args.strict, use_nvjpeg=args.gpu, batch_size=args.batch_size, verbose=args.verbose)
        assert not dcm.file_meta.TransferSyntaxUID.is_compressed
        dcm.save_as(dest)
    end_time = time()
    if args.verbose:
        total_time = end_time - start_time
        print(f"Total time (s): {total_time}")

    if args.test:
        with pydicom.dcmread(path) as dcm2:
            px1 = dcm.pixel_array
            px2 = dcm2.pixel_array
            assert (px1 == px2).all()


def entrypoint():
    parser = get_parser()
    main(parser.parse_args())


if __name__ == "__main__":
    entrypoint()

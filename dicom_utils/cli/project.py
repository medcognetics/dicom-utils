#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from argparse import ArgumentParser
from pathlib import Path
from typing import Union

import numpy as np
import pydicom

from ..dicom import convert_frame_voi_lut, decompress


def get_parser(parser: ArgumentParser = ArgumentParser()) -> ArgumentParser:
    parser.add_argument("path", type=Path, help="DICOM file to decompress")
    parser.add_argument("dest", type=Path, help="destination DICOM filepath")
    parser.add_argument("-m", "--method", default="max", choices=["max", "mean", "middle"], help="method of reduction")
    parser.add_argument(
        "-g", "--gpu", default=False, action="store_true", help="use NVJPEG2K accelerated decompression"
    )
    parser.add_argument(
        "-b", "--batch-size", default=4, type=int, help="batch size for NVJPEG2K accelerated decompression"
    )
    return parser


def safe_subtract(x: np.ndarray, other: Union[float, int, np.ndarray]) -> np.ndarray:
    return (x.astype(np.int64) - other).clip(min=0).astype(x.dtype)


def project(x: np.ndarray, reduction: str = "max") -> np.ndarray:
    VOLUME_DIM = -3
    if reduction == "max":
        x = x.max(axis=VOLUME_DIM)
    elif reduction == "mean":
        x = x.mean(axis=VOLUME_DIM).round().astype(x.dtype)
    elif reduction == "middle":
        L = x.shape[VOLUME_DIM]
        x = x[..., L // 2, :, :]
    return x


def main(args: argparse.Namespace) -> None:
    path = Path(args.path)
    dest = Path(args.dest)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not dest.parent.is_dir():
        raise NotADirectoryError(dest.parent)

    with pydicom.dcmread(path) as dcm:
        dcm = decompress(dcm, strict=False, use_nvjpeg=args.gpu, batch_size=args.batch_size)
        dcm = convert_frame_voi_lut(dcm)
        pixels = project(dcm.pixel_array, args.method)
        dcm.PixelData = pixels.tobytes()
        dcm.NumberOfFrames = 1
        dcm.save_as(dest)


def entrypoint():
    parser = get_parser()
    main(parser.parse_args())


if __name__ == "__main__":
    entrypoint()

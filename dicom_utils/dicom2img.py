#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm

from .dicom import read_image
from .types import Dicom


def get_parser(parser: ArgumentParser = ArgumentParser()) -> ArgumentParser:
    parser.add_argument("file", help="DICOM file to convert")
    parser.add_argument("dest", help="destination directory")
    parser.add_argument(
        "-s", "--split", default=False, action="store_true", help="split multi-frame inputs into separate files"
    )
    parser.add_argument("-f", "--fps", default=5, type=int, help="framerate for animated outputs")
    parser.add_argument("-q", "--quality", default=95, type=int, help="quality of outputs, from 1 to 100")
    return parser


def dicom_to_image(dcm: Dicom, dest: Path, split: bool = False, fps: int = 5, quality: int = 95) -> None:
    data = read_image(dcm)

    # min max norm
    min, max = data.min(), data.max()
    data = (data - min) / (max - min) * 255
    data = data.astype(np.uint8)

    # single image
    if dcm.pixel_array.ndim == 2:
        img = Image.fromarray(data)
        img.save(str(dest), quality=quality)

    elif split:
        subdir = Path(dest.with_suffix(""))
        subdir.mkdir(exist_ok=True)
        for i, frame in enumerate(data):
            path = Path(subdir, Path(f"{i}.png"))
            img = Image.fromarray(frame)
            img.save(str(path), quality=quality)

    else:
        frames = []
        for frame in data:
            path = Path()
            img = Image.fromarray(frame)
            frames.append(img)

        path = dest.with_suffix(".gif")
        duration_ms = len(frames) / (fps * 1000)
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, quality=quality)


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.DEBUG)
    pydicom.config.logger.setLevel(logging.DEBUG)
    for i in tqdm(range(10)):
        with pydicom.dcmread(args.file) as dcm:
            dicom_to_image(dcm, Path(args.dest), args.split, args.fps, args.quality)

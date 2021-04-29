#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from PIL import Image

from .dicom import read_image
from .types import Dicom


def get_parser(parser: ArgumentParser = ArgumentParser()) -> ArgumentParser:
    parser.add_argument("file", help="DICOM file to convert")
    parser.add_argument("-o", "--output", help="output filepath. if directory, use filename from `file`")
    parser.add_argument(
        "-s", "--split", default=False, action="store_true", help="split multi-frame inputs into separate files"
    )
    parser.add_argument("-f", "--fps", default=5, type=int, help="framerate for animated outputs")
    parser.add_argument("-q", "--quality", default=95, type=int, help="quality of outputs, from 1 to 100")
    return parser


def dicom_to_image(
    dcm: Dicom, dest: Optional[Path] = None, split: bool = False, fps: int = 5, quality: int = 95
) -> None:
    data = read_image(dcm)

    # min max norm
    min, max = data.min(), data.max()
    data = (data - min) / (max - min) * 255
    data = data.astype(np.uint8)
    H, W = data.shape[-2:]

    # single image
    if dcm.pixel_array.ndim == 2:
        if dest is None:
            plt.imshow(data.reshape(H, W), cmap="gray")
            print("Showing image")
            plt.show()
        else:
            img = Image.fromarray(data)
            img.save(str(dest), quality=quality)

    elif split:
        if dest is not None:
            subdir = Path(dest.with_suffix(""))
            subdir.mkdir(exist_ok=True)
        for i, frame in enumerate(data):
            if dest is not None:
                path = Path(subdir, Path(f"{i}.png"))
                img = Image.fromarray(frame)
                img.save(str(path), quality=quality)
            else:
                plt.imshow(frame, cmap="gray")
                print(f"Showing frame {i}/{len(data)}")
                plt.show()
                break

    else:
        frames = []
        for frame in data:
            img = Image.fromarray(frame)
            frames.append(img)

        duration_ms = len(frames) / (fps * 1000)
        if dest is None:
            raise NotImplementedError("3D inputs with no `dest` is not yet supported")
        else:
            path = dest.with_suffix(".gif")
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, quality=quality)


def main(args: argparse.Namespace) -> None:
    source = Path(args.file)
    dest = Path(args.output) if args.output is not None else None

    if not source.is_file():
        raise FileNotFoundError(source)

    # handle case where output path is a dir
    if dest is not None and dest.is_dir():
        dest = Path(dest, source.stem).with_suffix(".png")

    with pydicom.dcmread(source) as dcm:
        dicom_to_image(dcm, dest, args.split, args.fps, args.quality)


def entrypoint():
    parser = get_parser()
    main(parser.parse_args())


if __name__ == "__main__":
    entrypoint()

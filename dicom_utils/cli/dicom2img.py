#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterator, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from numpy import ndarray
from PIL import Image

from ..dicom import has_dicm_prefix, read_dicom_image
from ..logging import logger
from ..types import Dicom


def get_parser(parser: ArgumentParser = ArgumentParser()) -> ArgumentParser:
    parser.add_argument("path", help="DICOM file or folder with files to convert")
    parser.add_argument("-o", "--output", help="output filepath. if directory, use filename from `file`")
    parser.add_argument(
        "-s", "--split", default=False, action="store_true", help="split multi-frame inputs into separate files"
    )
    parser.add_argument("-f", "--fps", default=5, type=int, help="framerate for animated outputs")
    parser.add_argument("-q", "--quality", default=95, type=int, help="quality of outputs, from 1 to 100")
    parser.add_argument("--noblock", default=False, action="store_true", help="allow matplotlib to block")
    return parser


def to_collage(images: List[ndarray]) -> ndarray:
    num_images = len(images)

    assert num_images != 0, "There must be at least one image."
    assert all(len(i.shape) == 3 for i in images), "The images must have 3 dimensions."

    image_chns, max_image_rows, max_image_cols = np.array([i.shape for i in images]).max(axis=0)

    collage_rows = 1 if num_images < 3 else 2
    collage_cols = int(num_images / collage_rows + 0.5)

    collage = np.zeros((image_chns, collage_rows * max_image_rows, collage_cols * max_image_cols))

    for i, image in enumerate(images):
        row = int(i >= collage_cols)
        col = i % collage_cols
        start_row = row * max_image_rows
        start_col = col * max_image_cols
        image_chns, image_rows, image_cols = image.shape
        collage[:image_chns, start_row : start_row + image_rows, start_col : start_col + image_cols] = image

    return collage


def dcms_to_arrays(dcms: List[Dicom]) -> Iterator[ndarray]:
    for dcm in dcms:
        try:
            yield read_dicom_image(dcm)
        except Exception as e:
            logger.info(e)


def dcms_to_array(dcms: List[Dicom]) -> ndarray:
    return to_collage(list(dcms_to_arrays(dcms)))


def dicoms_to_graphic(
    dcms: List[Dicom],
    dest: Optional[Path] = None,
    split: bool = False,
    fps: int = 5,
    quality: int = 95,
    block: bool = True,
) -> None:
    data = dcms_to_array(dcms)

    # min max norm
    min, max = data.min(), data.max()
    data = (data - min) / (max - min) * 255
    data = data.astype(np.uint8)
    H, W = data.shape[-2:]

    # single image
    if dcms[0].pixel_array.ndim == 2:
        data = data[0]  # drop channel
        if dest is None:
            plt.imshow(data.reshape(H, W), cmap="gray")
            print("Showing image")
            plt.show(block=block)
        else:
            img = Image.fromarray(data)
            img.save(str(dest), quality=quality)

    elif split:
        if dest is not None:
            subdir = Path(dest.with_suffix(""))
            subdir.mkdir(exist_ok=True)
        else:
            subdir = None

        for i, frame in enumerate(data):
            if subdir is not None:
                path = Path(subdir, Path(f"{i}.png"))
                img = Image.fromarray(frame)
                img.save(str(path), quality=quality)
            else:
                plt.imshow(frame, cmap="gray")
                print(f"Showing frame {i}/{len(data)}")
                plt.show(block=block)
                break

    else:
        frames = [Image.fromarray(frame) for frame in data]
        duration_ms = len(frames) / (fps * 1000)
        if dest is None:
            raise NotImplementedError("3D inputs with no `dest` is not yet supported")
        else:
            path = dest.with_suffix(".gif")
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, quality=quality)


def path_to_sources(path: Path) -> List[Path]:
    if path.is_dir():
        return [f for f in path.iterdir() if has_dicm_prefix(f)]
    if path.is_file():
        return [path]
    else:
        raise FileNotFoundError(path)


def path_to_dicoms(path: Path) -> Iterator[Dicom]:
    for source in path_to_sources(path):
        try:
            yield pydicom.dcmread(source)
        except Exception as e:
            logger.info(e)


def main(args: argparse.Namespace) -> None:
    path = Path(args.path)
    dest = Path(args.output) if args.output is not None else None

    # handle case where output path is a dir
    if dest is not None and dest.is_dir():
        dest = Path(dest, path.stem).with_suffix(".png")

    dcms = list(path_to_dicoms(path))
    dicoms_to_graphic(dcms, dest, args.split, args.fps, args.quality, not args.noblock)


def entrypoint():
    parser = get_parser()
    main(parser.parse_args())


if __name__ == "__main__":
    entrypoint()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
from PIL import Image

from ..dicom import path_to_dicoms
from ..types import Dicom
from ..visualize import chw_to_hwc, dcms_to_annotated_images, to_collage


def get_parser(parser: ArgumentParser = ArgumentParser()) -> ArgumentParser:
    parser.add_argument("path", help="DICOM file or folder with files to convert")
    parser.add_argument("-o", "--output", help="output filepath. if directory, use filename from `file`")
    parser.add_argument("-d", "--downsample", help="downsample images by an integer factor", type=int)
    parser.add_argument(
        "-s", "--split", default=False, action="store_true", help="split multi-frame inputs into separate files"
    )
    parser.add_argument("-f", "--fps", default=5, type=int, help="framerate for animated outputs")
    parser.add_argument("-q", "--quality", default=95, type=int, help="quality of outputs, from 1 to 100")
    parser.add_argument("--noblock", default=False, action="store_true", help="allow matplotlib to block")
    parser.add_argument("--window", default=False, action="store_true", help="apply window from DICOM metadata")
    return parser


def dicoms_to_graphic(
    dcms: List[Dicom],
    dest: Optional[Path] = None,
    split: bool = False,
    fps: int = 5,
    quality: int = 95,
    block: bool = True,
    downsample: int = 1,
    **kwargs,
) -> None:
    images = dcms_to_annotated_images(dcms, **kwargs)
    data = to_collage([i.pixels[:, :, ::downsample, ::downsample] for i in images])

    if all(i.is_single_frame for i in images) or dest is None:
        data = chw_to_hwc(data[0])
        if dest is None:
            plt.imshow(data)
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
        frames = [Image.fromarray(chw_to_hwc(frame)) for frame in data]
        duration_ms = len(frames) / (fps * 1000)
        path = dest.with_suffix(".gif")
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, quality=quality)


def main(args: argparse.Namespace) -> None:
    path = Path(args.path)
    dest = Path(args.output) if args.output is not None else None

    # handle case where output path is a dir
    if dest is not None and dest.is_dir():
        dest = Path(dest, path.stem).with_suffix(".png")

    dcms = list(path_to_dicoms(path))
    dicoms_to_graphic(dcms, dest, args.split, args.fps, args.quality, not args.noblock, args.downsample, apply_window=args.window)


def entrypoint():
    parser = get_parser()
    main(parser.parse_args())


if __name__ == "__main__":
    entrypoint()

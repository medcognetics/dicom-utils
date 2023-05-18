#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import sys
from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ..dicom import path_to_dicoms
from ..types import Dicom
from ..visualize import DicomImage, chw_to_hwc, dcms_to_annotated_images, to_collage
from ..volume import VOLUME_HANDLERS, KeepVolume, VolumeHandler


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
    parser.add_argument("-b", "--bytes", default=False, action="store_true", help="output a png image as a byte stream")
    parser.add_argument(
        "-v",
        "--volume-handler",
        default="keep",
        choices=VOLUME_HANDLERS.available_keys(),
        help="volume handler for 3D inputs",
    )
    return parser


class ImageOutput:
    r"""Base class that describes how to render a given collage."""

    def __init__(self, images: List[DicomImage], quality: int = 95, downsample: int = 1, **kwargs):
        self.quality = quality
        self.images = images
        self.data = to_collage([i.pixels[:, :, ::downsample, ::downsample] for i in self.images])
        if self.is_single_frame:
            self.data = self.data[0]

    @property
    def chw_data(self) -> np.ndarray:
        return self.data

    @property
    def hwc_data(self) -> np.ndarray:
        return chw_to_hwc(self.data)

    @property
    def is_single_frame(self) -> bool:
        return all(i.is_single_frame for i in self.images)

    def to_bytes(self) -> bytes:
        img = Image.fromarray(self.hwc_data)
        buf = BytesIO()
        img.save(buf, format="png", quality=self.quality)
        return buf.getvalue()

    def print_bytes(self) -> None:
        sys.stdout.buffer.write(self.to_bytes())

    def save(self, path: Path) -> None:
        img = Image.fromarray(self.hwc_data)
        img.save(str(path), quality=self.quality)

    def show(self, block: bool = True, **kwargs) -> None:
        print("Showing image")
        plt.imshow(self.hwc_data, **kwargs)
        plt.show(block=block)

    @staticmethod
    def from_dicom_images(images: List[DicomImage], split: bool = False, **kwargs) -> "ImageOutput":
        if all(i.is_single_frame for i in images):
            return ImageOutput(images, **kwargs)
        elif split:
            return SplitImageOutput(images, **kwargs)
        else:
            return AnimatedImageOutput(images, **kwargs)


class SplitImageOutput(ImageOutput):
    r"""Renderer for 3D data that splits the 3D volume into multiple 2D frames"""

    def to_bytes(self) -> bytes:
        buf = BytesIO()
        for i, frame in enumerate(self.data):
            img = Image.fromarray(chw_to_hwc(frame))
            img.save(buf, format="png", quality=self.quality)
        return buf.getvalue()

    def save(self, path: Path) -> None:
        subdir = Path(path.with_suffix(""))
        subdir.mkdir(exist_ok=True)
        for i, frame in enumerate(self.data):
            path = Path(subdir, Path(f"{i}.png"))
            img = Image.fromarray(chw_to_hwc(frame))
            img.save(str(path), quality=self.quality)

    def show(self, **kwargs) -> None:
        for i, frame in enumerate(self.data):
            print(f"Showing frame {i}/{len(self.data)}")
            plt.imshow(chw_to_hwc(frame), cmap="gray")
            plt.show(block=True)


class AnimatedImageOutput(ImageOutput):
    r"""Renderer for 3D data that converts the volume to a 2D animation"""

    def __init__(self, images: List[DicomImage], quality: int = 95, downsample: int = 1, fps: int = 5):
        super().__init__(images, quality, downsample)
        self.fps = fps

    def _prepare_frames(self) -> List[Image.Image]:
        frames = [Image.fromarray(chw_to_hwc(frame)) for frame in self.data]
        return frames

    @property
    def duration_ms(self) -> float:
        return len(self.data) / (self.fps * 1000)

    def to_bytes(self) -> bytes:
        buf = BytesIO()
        frames = self._prepare_frames()
        frames[0].save(
            buf,
            save_all=True,
            append_images=frames[1:],
            duration=self.duration_ms,
            quality=self.quality,
            format="gif",
        )
        return buf.getvalue()

    def save(self, path: Path) -> None:
        frames = self._prepare_frames()
        path = path.with_suffix(".gif")
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=self.duration_ms, quality=self.quality)

    def show(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Showing an animated image is not yet implemented. "
            "Please save to an output GIF file or utilize piped byte stream"
        )


def dicoms_to_graphic(
    dcms: List[Dicom],
    dest: Optional[Path] = None,
    split: bool = False,
    fps: int = 5,
    quality: int = 95,
    block: bool = True,
    as_bytes: bool = False,
    downsample: int = 1,
    volume_handler: VolumeHandler = KeepVolume(),
    **kwargs,
) -> None:
    images = dcms_to_annotated_images(dcms, volume_handler=volume_handler, **kwargs)

    handler = ImageOutput.from_dicom_images(
        images,
        split,
        quality=quality,
        downsample=downsample,
        fps=fps,
    )

    if as_bytes:
        handler.print_bytes()
    elif dest is not None:
        handler.save(dest)
    else:
        handler.show(block)


def main(args: argparse.Namespace) -> None:
    path = Path(args.path)
    dest = Path(args.output) if args.output is not None else None

    # handle case where output path is a dir
    if dest is not None and dest.is_dir():
        dest = Path(dest, path.stem).with_suffix(".png")

    volume_handler = VOLUME_HANDLERS.get(args.volume_handler).instantiate_with_metadata().fn
    assert isinstance(volume_handler, VolumeHandler)

    dcms = list(path_to_dicoms(path))
    dicoms_to_graphic(
        dcms,
        dest,
        args.split,
        args.fps,
        args.quality,
        not args.noblock,
        args.bytes,
        args.downsample,
        volume_handler,
    )


def entrypoint():
    parser = get_parser()
    main(parser.parse_args())


if __name__ == "__main__":
    entrypoint()

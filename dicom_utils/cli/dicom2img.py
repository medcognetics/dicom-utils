#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterator, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from numpy import ndarray
from PIL import Image

from ..dicom import has_dicm_prefix, read_dicom_image
from ..logging import logger
from ..types import Dicom


Bbox = Tuple[int, int, int, int]
presentation_modality: Final[str] = "PR"


def rint(x: float) -> int:
    return int(round(x))


def dicom_ellipse_to_bbox(data: List[float]) -> Bbox:
    assert len(data) != 4 * 3, "This function is not implemented to support 3D (x,y,z) coordinates"
    assert len(data) == 4 * 2, f"Invalid number of data points ({len(data)}) for a DICOM ellipse"
    major_x0, major_y0, major_x1, major_y1, minor_x0, minor_y0, minor_x1, minor_y1 = data
    xs = [rint(v) for v in [major_x0, major_x1, minor_x0, minor_x1]]
    ys = [rint(v) for v in [major_y0, major_y1, minor_y0, minor_y1]]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return x0, y0, x1, y1


def dicom_circle_to_bbox(data: List[float]) -> Bbox:
    assert len(data) != 2 * 3, "This function is not implemented to support 3D (x,y,z) coordinates"
    assert len(data) == 2 * 2, f"Invalid number of data points ({len(data)}) for a DICOM circle"
    x_center, y_center, x_border, y_border = data
    radius = np.sqrt((x_center - x_border) ** 2 + (y_center - y_border) ** 2)
    x0, y0, x1, y1 = [rint(v) for v in [x_center - radius, y_center - radius, x_center + radius, y_center + radius]]
    return x0, y0, x1, y1


def dicom_trace_to_bbox(data: List[float], form: str) -> Bbox:
    if form == "CIRCLE":
        return dicom_circle_to_bbox(data)
    elif form == "ELLIPSE":
        return dicom_ellipse_to_bbox(data)
    else:
        raise Exception(f"Drawing is not supported for {form}")


def chw_to_hwc(image: ndarray) -> ndarray:
    """CxHxW -> HxWxC"""
    return np.rollaxis(image, 0, 3).copy()


def hwc_to_chw(image: ndarray) -> ndarray:
    """HxWxC -> CxHxW"""
    return np.rollaxis(image, 2, 0).copy()


def draw_bbox(image: ndarray, bbox: Bbox, color: Tuple[float, float, float] = (0, 1, 0), thickness: int = 30) -> None:
    x0, y0, x1, y1 = bbox

    for i, slice in enumerate(image):
        slice = chw_to_hwc(slice)
        cv2.rectangle(slice, (x0, y0), (x1, y1), color, thickness)
        image[i] = hwc_to_chw(slice)


@dataclass
class Annotation:
    """Store an annotation with corresponding DICOM filename"""

    def __init__(self, sop_uid: str, data: List[float], form: str):
        self.uid = sop_uid
        self.form = form
        self.bbox = dicom_trace_to_bbox(data, self.form)

    def __repr__(self):
        return f"<SOPInstanceUID: {self.uid}, form: {self.form}, bbox: {self.bbox}>"


@dataclass
class DicomImage:
    """Store DICOM image pixels with associated metadata"""

    pixels: ndarray
    uid: str

    @classmethod
    def from_dicom(cls, dicom: Dicom) -> "DicomImage":
        pixels = read_dicom_image(dicom)
        pixels = to_rgb(pixels)
        return cls(pixels, dicom.SOPInstanceUID)

    @property
    def is_single_slice(self) -> bool:
        return self.pixels.shape[0] == 1

    def __repr__(self):
        return f"<SOPInstanceUID: {self.uid}, shape: {self.pixels.shape}>"


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
    assert all(len(i.shape) == 4 for i in images), "The images must have 4 dimensions."

    image_chns, _, max_image_rows, max_image_cols = np.array([i.shape for i in images]).max(axis=0)

    collage_rows = 1 if num_images < 3 else 2
    collage_cols = int(num_images / collage_rows + 0.5)

    assert all(i.shape[1] == 3 for i in images)
    collage = np.zeros((image_chns, 3, collage_rows * max_image_rows, collage_cols * max_image_cols))

    for i, image in enumerate(images):
        row = int(i >= collage_cols)
        col = i % collage_cols
        start_row = row * max_image_rows
        start_col = col * max_image_cols
        image_chns, _, image_rows, image_cols = image.shape
        collage[:image_chns, :, start_row : start_row + image_rows, start_col : start_col + image_cols] = image

    return collage


def dcms_to_images(dcms: List[Dicom]) -> Iterator[DicomImage]:
    for dcm in dcms:
        try:
            yield DicomImage.from_dicom(dcm)
        except Exception as e:
            logger.info(e)


def dcm_to_annotations(dcm: Dicom) -> Iterator[Annotation]:
    if "GraphicAnnotationSequence" in dcm.dir():
        for graphic in dcm.GraphicAnnotationSequence:
            if "GraphicObjectSequence" in graphic.dir():
                sop_uid = graphic.ReferencedImageSequence[0].ReferencedSOPInstanceUID
                data = graphic.GraphicObjectSequence[0].GraphicData
                shape = graphic.GraphicObjectSequence[0].GraphicType
                yield Annotation(sop_uid, data, shape)


def dcms_to_annotations(dcms: List[Dicom]) -> Iterator[Annotation]:
    for dcm in [d for d in dcms if d.Modality == presentation_modality]:
        try:
            yield from dcm_to_annotations(dcm)
        except Exception as e:
            logger.info(e)


def overlay_annotations(images: List[DicomImage], annotations: List[Annotation]) -> None:
    for image in images:
        for annotation in annotations:
            if annotation.uid == image.uid:
                draw_bbox(image.pixels, annotation.bbox)


def to_rgb(image: ndarray) -> ndarray:
    chns, rows, cols = image.shape
    rgb = np.zeros((chns, 3, rows, cols))

    for i, channel in enumerate(image):
        norm_pixels = channel / channel.max()

        for j in range(3):
            rgb[i, j, :, :] = norm_pixels

    return rgb


def dcms_to_annotated_images(dcms: List[Dicom]) -> List[DicomImage]:
    images = list(dcms_to_images(dcms))
    annotations = list(dcms_to_annotations(dcms))
    overlay_annotations(images, annotations)
    return images


def to_8bit(x: ndarray) -> ndarray:
    min, max = x.min(), x.max()
    x = (x - min) / (max - min) * 255
    return x.astype(np.uint8)


def dicoms_to_graphic(
    dcms: List[Dicom],
    dest: Optional[Path] = None,
    split: bool = False,
    fps: int = 5,
    quality: int = 95,
    block: bool = True,
) -> None:
    images = dcms_to_annotated_images(dcms)
    data = to_8bit(to_collage([i.pixels for i in images]))

    if all(i.is_single_slice for i in images):
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

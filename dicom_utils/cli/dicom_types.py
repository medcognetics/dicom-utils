#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from argparse import ArgumentParser
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import pydicom

from ..dicom import has_dicm_prefix
from ..types import ImageType


def get_parser(parser: ArgumentParser = ArgumentParser()) -> ArgumentParser:
    parser.add_argument("path", default="./", help="path to find from")
    parser.add_argument("--name", "-n", default="*", help="glob pattern for filename")
    parser.add_argument("--jobs", "-j", default=4, help="parallelism")
    parser.add_argument(
        "--unique", "-u", default=False, action="store_true", help="only print the first match for a unique image type"
    )
    parser.add_argument("--include-numeric", default=False, action="store_true", help="include numeric fields")
    # TODO add a field to only return DICOMs with readable image data
    return parser


def check_file(path: Path, args: argparse.Namespace) -> Optional[Tuple[Path, int, str, str]]:
    if not path.is_file() or not has_dicm_prefix(path):
        return
    num_frames = 1
    try:
        tags = [0x00080008, "NumberOfFrames", 0x00081090]
        with pydicom.dcmread(path, specific_tags=tags, stop_before_pixels=True) as dcm:
            image_type = dcm.get(0x00080008, None)
            if not image_type:
                return
            image_type = image_type.value
            if isinstance(image_type, str):
                image_type = [image_type]
            img_type = ImageType.from_dicom(dcm)
            simple_image_type = img_type.to_simple_image_type()
            num_frames = img_type.NumberOfFrames or 1
    except AttributeError:
        return

    final_image_type: List[Optional[str]] = [None]
    return path, num_frames, str(simple_image_type), img_type.simple_repr()


def main(args: argparse.Namespace) -> None:
    path = Path(args.path)
    if not path.is_dir():
        raise NotADirectoryError(path)

    seen_types = set()

    def callback(x: Future):
        result = x.result()
        if result is None:
            return
        path, num_frames, simple_type, image_type = result

        if image_type not in seen_types:
            print(f"{path} - {num_frames} - {simple_type} - {image_type}", flush=True)
            seen_types.add(image_type)

    futures: List[Future] = []
    with ThreadPoolExecutor(args.jobs) as tp:
        for match in path.rglob(args.name):
            f = tp.submit(check_file, match, args)
            f.add_done_callback(callback)
            futures.append(f)


def entrypoint():
    parser = get_parser()
    main(parser.parse_args())


if __name__ == "__main__":
    entrypoint()

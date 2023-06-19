#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
from argparse import ArgumentParser
from hashlib import md5
from pathlib import Path
from typing import Optional, Tuple

import pydicom
from tqdm import tqdm
from tqdm_multiprocessing import ConcurrentMapper

from ..container.collection import iterate_input_path
from ..dicom import decompress as decompress_fn
from ..dicom import has_dicm_prefix
from ..tags import Tag
from ..types import get_value


def get_parser(parser: ArgumentParser = ArgumentParser()) -> ArgumentParser:
    parser.prog = "hash"
    parser.description = "Compute the hash of one or more DICOM files"
    parser.add_argument("target", default="./", help="path to find from")
    parser.add_argument("--tag", "-t", default="PixelData", help="Tag to compute hash on")
    parser.add_argument(
        "--decompress",
        "-d",
        default=False,
        action="store_true",
        help="When tag=PixelData, decompress the data before hashing",
    )
    parser.add_argument("--jobs", "-j", default=4, help="Parallelism")
    return parser


def compute_hash(path: Path, tag: Tag = Tag.PixelData, decompress: bool = False) -> Optional[Tuple[Path, str]]:
    try:
        if not path.is_file() or not has_dicm_prefix(path):
            return
        ds = pydicom.dcmread(path)
        if tag == Tag.PixelData and decompress:
            ds = decompress_fn(ds)
        value = get_value(ds, tag, "")
        return path, md5(value).hexdigest() if isinstance(value, bytes) else md5(value.encode()).hexdigest()
    except Exception as e:
        logging.exception(f"Failed to hash {path}", exc_info=e)


def main(args: argparse.Namespace) -> None:
    src = Path(args.target)
    sources = [p for p in iterate_input_path(src) if p.is_file()]
    tag = getattr(Tag, args.tag)
    with ConcurrentMapper(jobs=int(args.jobs)) as mapper:
        mapper.create_bar("Hashing", total=len(sources))
        tqdm.write("path,hash")
        for result in mapper(compute_hash, sources, tag=tag, decompress=args.decompress):
            if result is not None:
                path, hashval = result
                tqdm.write(f"{path},{hashval}")


def entrypoint():
    parser = get_parser()
    main(parser.parse_args())


if __name__ == "__main__":
    entrypoint()

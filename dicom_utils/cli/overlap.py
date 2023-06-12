#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from argparse import ArgumentParser
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Set

from ..tags import Tag
from .hash import compute_hash


def get_parser(parser: ArgumentParser = ArgumentParser()) -> ArgumentParser:
    parser.add_argument("path1", help="first path to search")
    parser.add_argument("path2", help="second path to search")
    parser.add_argument("--name", "-n", default="*", help="glob pattern for filename")
    parser.add_argument("--parents", "-p", default=True, action="store_true", help="return unique parent directories")
    parser.add_argument("--jobs", "-j", default=4, help="parallelism")
    parser.add_argument("--tag", "-t", default="PixelData", help="Tag to check overlap on")
    parser.add_argument(
        "--decompress",
        "-d",
        default=False,
        action="store_true",
        help="When tag=PixelData, decompress the data before hashing",
    )
    # TODO add a field to only return DICOMs with readable image data
    return parser


def main(args: argparse.Namespace) -> None:
    path1 = Path(args.path1)
    path2 = Path(args.path2)
    if not path1.is_dir():
        raise NotADirectoryError(path1)
    if not path2.is_dir():
        raise NotADirectoryError(path2)

    path1_seen: Dict[str, Set[Path]] = {}
    path2_seen: Dict[str, Set[Path]] = {}

    def callback(x: Future):
        result = x.result()
        if result is None:
            return
        path, hashval = result

        if path.is_relative_to(args.path1):
            container = path1_seen
        elif path.is_relative_to(args.path2):
            container = path2_seen
        else:
            raise RuntimeError()

        seen = container.get(hashval, set())
        seen.add(path)
        container[hashval] = seen

    futures: List[Future] = []
    tag = getattr(Tag, args.tag)
    with ThreadPoolExecutor(int(args.jobs)) as tp:
        for match in path1.rglob(args.name):
            f = tp.submit(compute_hash, match, tag=tag, decompress=args.decompress)
            f.add_done_callback(callback)
            futures.append(f)
        for match in path2.rglob(args.name):
            f = tp.submit(compute_hash, match, tag=tag, decompress=args.decompress)
            f.add_done_callback(callback)
            futures.append(f)

    for hashval, pathset1 in path1_seen.items():
        pathset2 = path2_seen.get(hashval, {})

        if not pathset2:
            continue

        for p in pathset1:
            print(f"{p},{hashval}")
        for p in pathset2:
            print(f"{p},{hashval}")


def entrypoint():
    parser = get_parser()
    main(parser.parse_args())


if __name__ == "__main__":
    entrypoint()

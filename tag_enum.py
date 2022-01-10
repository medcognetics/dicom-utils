#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Script for generating an IntEnum of DICOM tags from pydicom's dictionary"""
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Final, List

from pydicom.datadict import DicomDictionary


MARKER: Final = "# >>> BEGIN TAGS >>>"


def write(path: Path) -> None:
    # read methods up to start of tag enum values
    lines: List[str] = []
    with open(path, "r") as f:
        for line in f.readlines():
            lines.append(line)
            if line.strip() == MARKER:
                break

    # write tag enum values
    with open(path, "w") as f:
        f.writelines(lines)
        for tag, v in DicomDictionary.items():
            keyword = v[-1]
            if keyword:
                f.write(f"    {keyword} = {tag}\n")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate IntEnum of DICOM tags")
    parser.add_argument("dest", help="Output filepath")
    return parser.parse_args()


def main(args: Namespace):
    dest = Path(args.dest)
    if not dest.parent.is_dir():
        raise NotADirectoryError(dest.parent)
    write(dest)


if __name__ == "__main__":
    main(parse_args())

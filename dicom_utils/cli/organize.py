#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser, Namespace
from os import PathLike
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

from tqdm import tqdm

from ..container.collection import RecordCollection
from ..container.group import GROUP_REGISTRY
from ..container.input import Input
from ..container.output import OUTPUT_REGISTRY
from ..container.record import HELPER_REGISTRY


def organize(
    sources: Union[PathLike, Iterable[PathLike]],
    dest: PathLike,
    records: Optional[Iterable[str]] = None,
    groups: Iterable[str] = ["patient-id", "study-uid"],
    helpers: Iterable[str] = [],
    namers: Iterable[str] = ["patient-id", "study-uid"],
    outputs: Iterable[str] = ["symlink-cases"],
    use_bar: bool = True,
    jobs: Optional[int] = None,
    threads: bool = False,
    timeout: Optional[int] = None,
    **kwargs,
) -> Dict[str, Dict[str, RecordCollection]]:
    inp = Input(
        sources,
        records,
        groups,
        helpers,
        namers,
        use_bar=use_bar,
        jobs=jobs,
        threads=threads,
        timeout=timeout,
        **kwargs,
    )

    result: Dict[str, Dict[str, RecordCollection]] = {}
    for output_name in tqdm(outputs, desc="Writing outputs", disable=not use_bar):
        output = OUTPUT_REGISTRY.get(output_name).instantiate_with_metadata(
            root=dest,
            use_bar=use_bar,
            jobs=jobs,
            threads=True,
            timeout=timeout,
        )
        result[output_name] = output(inp)

    return result


def get_parser(parser: ArgumentParser = ArgumentParser()) -> ArgumentParser:
    parser.add_argument("paths", nargs="+", type=Path, help="path to source files")
    parser.add_argument("dest", type=Path, help="path to outputs")
    parser.add_argument(
        "-g",
        "--group",
        default="study_uid",
        choices=GROUP_REGISTRY.available_keys(),
        help="grouping function",
    )

    parser.add_argument(
        "-o",
        "--output",
        nargs="+",
        default=list(OUTPUT_REGISTRY.available_keys()),
        choices=OUTPUT_REGISTRY.available_keys(),
        help="output functions",
    )

    parser.add_argument(
        "--helpers",
        nargs="+",
        default=[],
        choices=HELPER_REGISTRY.available_keys(),
        help="helper functions",
    )

    parser.add_argument("-m", "--modality", default=None, help="modality override")
    parser.add_argument("-j", "--jobs", default=8, type=int, help="number of parallel jobs")
    parser.add_argument(
        "--allow-non-dicom", default=False, action="store_true", help="keep groups that don't include a DICOM file"
    )
    parser.add_argument(
        "-n", "--numbering-start", default=1, type=int, help="start of numbering for output case symlinks"
    )
    parser.add_argument(
        "--is-sfm", default=False, action="store_true", help="target DICOM files are SFM (as opposed to FFDM)"
    )
    return parser


def main(args: Namespace):
    for p in args.paths:
        if not p.is_dir():
            raise NotADirectoryError(p)
    if not args.dest.is_dir():
        raise NotADirectoryError(args.dest)

    organize(
        args.paths,
        args.dest,
        groups=[args.group],
        outputs=args.output,
        helpers=args.helpers,
        jobs=args.jobs,
        modality=args.modality,
        require_dicom=not args.allow_non_dicom,
    )


def entrypoint():
    parser = get_parser()
    main(parser.parse_args())


if __name__ == "__main__":
    entrypoint()

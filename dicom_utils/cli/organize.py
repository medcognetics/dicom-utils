#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser, Namespace
from os import PathLike
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from tqdm import tqdm

from ..container.collection import RecordCollection
from ..container.group import GROUP_REGISTRY
from ..container.input import Input
from ..container.output import OUTPUT_REGISTRY, Output
from ..container.record import HELPER_REGISTRY


def organize(
    sources: Union[PathLike, Iterable[PathLike]],
    dest: PathLike,
    records: Optional[Iterable[str]] = None,
    groups: Iterable[str] = ["patient-id", "study-date", "study-uid"],
    helpers: Iterable[str] = [],
    namers: Iterable[str] = ["patient-id", "study-date", "study-uid"],
    outputs: Iterable[str] = ["symlink-cases"],
    use_bar: bool = True,
    **kwargs,
) -> Dict[Output, Dict[str, RecordCollection]]:
    inp = Input(sources, records, groups, helpers, namers, use_bar=use_bar, **kwargs)
    opt: List[Output] = []
    derived_opt: List[Output] = []
    for o in outputs:
        reg = OUTPUT_REGISTRY.get_with_metadata(o)
        subdir = Path(dest, str(reg.metadata.get("subdir", "")))
        fn = reg.fn
        derived = reg.metadata.get("derived", False)
        (derived_opt if derived else opt).append(fn(path=subdir))

    result: Dict[Output, Dict[str, RecordCollection]] = {}
    for o in tqdm(opt, desc="Writing outputs", disable=not use_bar):
        result[o] = o(inp)
    for o in tqdm(derived_opt, desc="Writing derived outputs", disable=not use_bar):
        result[o] = o(next(iter(result.values())))
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

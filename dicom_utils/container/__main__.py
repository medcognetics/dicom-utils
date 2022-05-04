#!/usr/bin/env python
# -*- coding: utf-8 -*-


from argparse import ArgumentParser, Namespace
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union, cast

from tqdm import tqdm

from .collection import RecordCollection
from .group import GROUP_REGISTRY
from .input import Input
from .output import OUTPUT_REGISTRY, Output
from .record import HELPER_REGISTRY


def run(
    sources: Union[PathLike, Iterable[PathLike]],
    dest: PathLike,
    records: Optional[Iterable[str]] = None,
    groups: Iterable[str] = ["study-uid"],
    helpers: Iterable[str] = [],
    outputs: Iterable[str] = ["symlink-cases", "longitudinal"],
    prefix: str = "Case-",
    start: int = 1,
    use_bar: bool = True,
    **kwargs,
) -> Dict[Output, Dict[str, RecordCollection]]:
    inp = Input(sources, records, groups, helpers, prefix, start, use_bar=use_bar, **kwargs)
    opt: List[Output] = []
    derived_opt: List[Output] = []
    for o in outputs:
        reg_dict = cast(Dict[str, Any], OUTPUT_REGISTRY.get(o, with_metadata=True))
        subdir = Path(dest, str(reg_dict["metadata"].get("subdir", "")))
        fn = reg_dict["fn"]
        derived = reg_dict["metadata"].get("derived", False)
        (derived_opt if derived else opt).append(fn(subdir))

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
        "-g", "--group", default="study_uid", choices=GROUP_REGISTRY.available_keys(), help="grouping function"
    )
    parser.add_argument(
        "-o",
        "--output",
        nargs="+",
        default=["symlink-cases", "symlink-mammograms", "symlink-complete-mammograms", "longitudinal"],
        choices=OUTPUT_REGISTRY.available_keys(),
        help="output functions",
    )

    parser.add_argument(
        "--helpers", nargs="+", default=[], choices=HELPER_REGISTRY.available_keys(), help="helper functions"
    )

    parser.add_argument("-a", "--annotation", type=Path, default=None, help="path to annotation files")
    parser.add_argument("-p", "--prefix", default="MedCog-", help="prefix for symlinked cases")
    parser.add_argument("-j", "--jobs", default=8, type=int, help="number of parallel jobs")
    parser.add_argument(
        "-k", "--keep-duplicates", default=False, action="store_true", help="keep images with dulicate UIDs"
    )
    parser.add_argument(
        "-n", "--numbering-start", default=1, type=int, help="start of numbering for output case symlinks"
    )
    parser.add_argument(
        "--is-sfm", default=False, action="store_true", help="target DICOM files are SFM (as opposed to FFDM)"
    )
    parser.add_argument(
        "-m",
        "--merge-by-dir",
        default=False,
        action="store_true",
        help="merge files with a common parent directory regardless of StudyInstanceUID",
    )
    parser.add_argument(
        "-y", "--year-offset", default=None, type=int, help="filepath offset for parsing year information"
    )
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="verbosely report errors")
    return parser


def main(args: Namespace):
    for p in args.paths:
        if not p.is_dir():
            raise NotADirectoryError(p)
    if not args.dest.is_dir():
        raise NotADirectoryError(args.dest)

    result = run(args.paths, args.dest, groups=[args.group], outputs=args.output, helpers=args.helpers)
    print(len(result))


if __name__ == "__main__":
    main(get_parser().parse_args())

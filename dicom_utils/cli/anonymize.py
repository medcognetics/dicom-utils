import argparse
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, List, Optional

import pydicom
from tqdm_multiprocessing import ConcurrentMapper

from ..anonymize import anonymize
from ..container import iterate_input_path
from ..dicom import has_dicm_prefix


def _anonymize(path: Path, output: Path, root: Optional[Path]) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)  # pragma: no cover

    dest_path = output / (path.relative_to(root) if root is not None else path.name)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".dcm" or has_dicm_prefix(path):
        with pydicom.dcmread(path) as dcm:
            anonymize(dcm)
            dcm.save_as(dest_path)
    else:
        shutil.copy(path, dest_path)

    return dest_path


def batch_anonymize(
    paths: Iterable[Path],
    output: Path,
    root: Optional[Path],
    **kwargs,
) -> List[Path]:
    """
    Anonymizes a batch of DICOM files.

    Args:
        paths: The paths of the DICOM files to be anonymized.
        output: The output directory where the anonymized DICOM files will be saved.
        root: The root directory for naming output files. If not specified only the output filename will be used.

    Keyword Args:
        Forwards keyword arguments to ConcurrentMapper.

    Returns:
        The paths of the anonymized DICOM files.
    """
    if not output.is_dir():
        raise NotADirectoryError(output)  # pragma: no cover

    paths = list(paths)
    with ConcurrentMapper(**kwargs) as mapper:
        mapper.create_bar(total=len(paths), desc="Anonymizing")
        result = list(
            mapper(
                _anonymize,
                paths,
                output=output,
                root=root,
            )
        )
    return result


def get_parser(parser: ArgumentParser = ArgumentParser()) -> ArgumentParser:
    parser.add_argument(
        "input", type=Path, help="Anonymization targets. May be a DICOM file, directory, or text file with paths."
    )
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument(
        "-r",
        "--root",
        type=Path,
        default=None,
        help=(
            "Root directory for naming output files. Use this to retain parent directories of the input file in the output. "
            "If not specified only the output filename will be used."
        ),
    )
    parser.add_argument("-j", "--jobs", type=int, default=None, help="Number of parallel jobs to run.")
    return parser


def main(args: argparse.Namespace) -> None:
    batch_anonymize(iterate_input_path(args.input), args.output, args.root)


def entrypoint():
    parser = get_parser()
    main(parser.parse_args())


if __name__ == "__main__":
    entrypoint()

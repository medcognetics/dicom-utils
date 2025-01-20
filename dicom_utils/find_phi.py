import copy
import difflib
import os
from typing import Iterator, List

import pydicom
from colorama import Fore
from pydicom import Dataset
from tqdm import tqdm

from dicom_utils.dicom import has_dicm_prefix

from .anonymize import anonymize


def color_diff(diff):
    for line in diff:
        if line.startswith("+"):
            yield Fore.GREEN + line + Fore.RESET
        elif line.startswith("-"):
            yield Fore.RED + line + Fore.RESET
        elif line.startswith("?"):
            yield Fore.BLUE + line + Fore.RESET
        else:
            pass


def gen_dicoms(path: str) -> Iterator[str]:
    for root, folders, filenames in os.walk(path):
        for f in filenames:
            filename = os.path.join(root, f)
            if has_dicm_prefix(filename):
                yield filename


def dataset_to_str(ds: Dataset) -> List[str]:
    return str(ds).splitlines(keepends=True)


def find_phi(path: str, overwrite: bool, verbose: bool) -> None:
    filenames = [path] if os.path.isfile(path) else list(gen_dicoms(path))

    for filename in tqdm(filenames):
        ds = pydicom.dcmread(filename)
        str(ds)  # I think this forces evaluation of certain fields. Without this, the anonymizer may throw an error.
        ds_str = dataset_to_str(copy.deepcopy(ds))
        anonymize(ds)

        if verbose:
            print(filename)
            for diff in color_diff(difflib.ndiff(ds_str, dataset_to_str(ds))):
                print(diff)

        if overwrite:
            ds.save_as(filename)

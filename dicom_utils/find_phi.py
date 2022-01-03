import copy
import difflib
import os
from typing import Final, Iterator, List, Tuple

import pydicom
from colorama import Fore
from pydicom import Dataset
from tqdm import tqdm

from dicom_utils.dicom import has_dicm_prefix

from .anonymize import anonymize


Tag = Tuple[str, str]

number_of_frames: Final[Tag] = ("0028", "0008")
irradiation_event_uid: Final[Tag] = ("0008", "3010")
fake_uid: Final[bytes] = b"0.0.000.000000.0000.00.0000000000000000.00000000000.00000000"
uid_len: Final[int] = len(fake_uid)


def fix_bad_fields(raw_elem, **kwargs):
    if raw_elem.tag == number_of_frames and raw_elem.value is None:
        # Value of "None" is non-conformant
        raw_elem = raw_elem._replace(value=1, length=1)
    elif raw_elem.tag == irradiation_event_uid and len(raw_elem.value) > uid_len:
        # The DICOM anonymizer doesn't handle a list of UIDs properly
        raw_elem = raw_elem._replace(value=fake_uid, length=uid_len)

    return raw_elem


pydicom.config.data_element_callback = fix_bad_fields  # type: ignore
pydicom.config.convert_wrong_length_to_UN = True  # type: ignore


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

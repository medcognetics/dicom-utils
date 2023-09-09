import copy
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pydicom
from pydicom import Dataset
from pydicom.errors import InvalidDicomError
from pydicom.uid import generate_uid


def merge_datasets(datasets: List[Dataset]) -> Dataset:
    new_dataset = copy.copy(datasets[0])

    if len(datasets) > 1:
        new_dataset.SOPInstanceUID = generate_uid()
        new_dataset.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ordered_by_instance_num = sorted(datasets, key=lambda x: int(x.InstanceNumber))
        frames = [ds.PixelData for ds in ordered_by_instance_num]
        new_dataset.PixelData = b"".join(frames)
        new_dataset.NumberOfFrames = len(frames)

        # TODO Should we delete other fields from `new_dataset`?
        if "InstanceNumber" in new_dataset:
            del new_dataset["InstanceNumber"]

    return new_dataset


def create_merged_datasets(path: Path) -> List[Dataset]:
    series_lookup: Dict[str, Any] = defaultdict(list)

    for file in path.rglob("*"):
        try:
            ds = pydicom.dcmread(file)
            series_lookup[ds.SeriesInstanceUID].append(ds)
        except InvalidDicomError as e:
            logging.exception(f"Failed to open `{file}`", exc_info=e)

    merged_datasets = [merge_datasets(datasets) for datasets in series_lookup.values()]
    return merged_datasets


def merge_series(path: Path) -> None:
    # Sometimes each frame for a 3D image is stored in a separate DICOM.
    # This code merges the separate frame files into a single DICOM file containing the 3D image.
    for i, dataset in enumerate(create_merged_datasets(path)):
        dataset.save_as(path / f"merged-{i}.dcm")

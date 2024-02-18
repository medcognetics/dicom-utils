import shutil
from pathlib import Path

import pytest


@pytest.fixture
def dicom_folder(dicom_file, tmp_path):
    parent = Path(tmp_path)
    parent.mkdir(exist_ok=True)
    file_list = []
    for i in range(3):
        child = Path(dicom_file).with_suffix("").stem
        child = Path(parent, f"{child}_{i}").with_suffix(".dcm")
        shutil.copy(dicom_file, child)
        file_list.append(child)

    return file_list

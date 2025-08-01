import runpy
import shutil
import sys
from pathlib import Path
from typing import Final

import pydicom
import pytest

from dicom_utils.anonymize import is_anonymized


num_dicom_test_files: Final[int] = 3


@pytest.fixture(params=pydicom.data.get_testdata_files("*rgb*.dcm")[:num_dicom_test_files])  # type: ignore
def test_dicom(tmp_path, request) -> Path:
    path = tmp_path / "file.dcm"
    shutil.copy(request.param, path)
    path.with_name("report.json").touch()
    return path


def test_anonymize_dicom(tmp_path, test_dicom) -> None:
    dest = tmp_path / "output"
    dest.mkdir()
    sys.argv = [sys.argv[0], str(test_dicom.parent), str(dest), "-j", "4", "-r", str(tmp_path)]
    runpy.run_module("dicom_utils.cli.anonymize", run_name="__main__", alter_sys=True)

    assert list(dest.rglob("*.dcm"))
    for path in dest.rglob("*.dcm"):
        with pydicom.dcmread(path) as dcm:
            assert is_anonymized(dcm)


def test_anonymize_json(tmp_path, test_dicom) -> None:
    dest = tmp_path / "output"
    dest.mkdir()
    sys.argv = [sys.argv[0], str(test_dicom.parent), str(dest), "-j", "4", "-r", str(tmp_path)]
    runpy.run_module("dicom_utils.cli.anonymize", run_name="__main__", alter_sys=True)
    assert list(dest.rglob("*.json"))

#!/usr/bin/env python
# -*- coding: utf-8 -*-


import runpy
import shutil
import sys
from pathlib import Path

import pytest


@pytest.fixture
def dicom_file():
    pydicom = pytest.importorskip("pydicom")
    return pydicom.data.get_testdata_file("CT_small.dcm")


@pytest.fixture
def other_dicom_file():
    pydicom = pytest.importorskip("pydicom")
    return pydicom.data.get_testdata_file("MR_small.dcm")


def test_series_duplicate(dicom_file, capsys, tmp_path):
    study_path = Path(tmp_path, "study")
    series_1 = Path(study_path, "file1.dcm")
    series_2 = Path(study_path, "file2.dcm")
    series_1.parent.mkdir(parents=True)
    series_2.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(dicom_file, series_1)
    shutil.copy(dicom_file, series_2)

    sys.argv = [sys.argv[0], str(tmp_path), "series"]
    runpy.run_module("dicom_utils.cli.duplicate", run_name="__main__", alter_sys=True)
    captured = capsys.readouterr()

    # order is non-deterministic
    expected1 = f"{series_1}\t{series_2}"
    expected2 = f"{series_2}\t{series_1}"
    assert expected1 in captured.out or expected2 in captured.out


def test_series_non_duplicate(dicom_file, other_dicom_file, capsys, tmp_path):
    study_path = Path(tmp_path, "study")
    series_1 = Path(study_path, "file1.dcm")
    series_2 = Path(study_path, "file2.dcm")
    series_1.parent.mkdir(parents=True)
    series_2.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(dicom_file, series_1)
    shutil.copy(other_dicom_file, series_2)

    sys.argv = [sys.argv[0], str(tmp_path), "series"]
    runpy.run_module("dicom_utils.cli.duplicate", run_name="__main__", alter_sys=True)
    captured = capsys.readouterr()

    expected1 = f"{series_1}\t{series_2}"
    expected2 = f"{series_2}\t{series_1}"
    assert expected1 not in captured.out and expected2 not in captured.out


def test_study_duplicate(dicom_file, capsys, tmp_path):
    series_1 = Path(tmp_path, "study1", "file1.dcm")
    series_2 = Path(tmp_path, "study2", "file1.dcm")
    series_1.parent.mkdir(parents=True)
    series_2.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(dicom_file, series_1)
    shutil.copy(dicom_file, series_2)

    sys.argv = [sys.argv[0], str(tmp_path), "study"]
    runpy.run_module("dicom_utils.cli.duplicate", run_name="__main__", alter_sys=True)
    captured = capsys.readouterr()

    # order is non-deterministic
    expected1 = f"{series_1.parent}\t{series_2.parent}"
    expected2 = f"{series_2.parent}\t{series_1.parent}"
    assert expected1 in captured.out or expected2 in captured.out

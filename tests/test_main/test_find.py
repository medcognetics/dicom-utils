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


def test_find(dicom_folder, capsys, tmp_path):
    sys.argv = [
        sys.argv[0],
        str(tmp_path),
    ]
    runpy.run_module("dicom_utils.cli.find", run_name="__main__", alter_sys=True)
    captured1 = capsys.readouterr()
    runpy.run_module("dicom_utils.cli.find", run_name="__main__", alter_sys=True)
    captured2 = capsys.readouterr()

    for p in dicom_folder:
        assert str(p) in captured1.out
    assert list(sorted(captured1)) == list(sorted(captured2))


@pytest.mark.usefixtures("dicom_folder")
def test_find_pattern(capsys, tmp_path):
    sys.argv = [sys.argv[0], str(tmp_path), "--name", "*.DCM"]
    runpy.run_module("dicom_utils.cli.find", run_name="__main__", alter_sys=True)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_find_parents(dicom_folder, capsys, tmp_path):
    sys.argv = [sys.argv[0], str(tmp_path), "--parents"]
    runpy.run_module("dicom_utils.cli.find", run_name="__main__", alter_sys=True)
    captured = capsys.readouterr()
    for p in dicom_folder:
        assert str(p.parent) in captured.out

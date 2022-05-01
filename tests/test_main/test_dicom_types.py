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


def test_dicom_types(dicom_folder, capsys, tmp_path):
    sys.argv = [
        sys.argv[0],
        str(tmp_path),
    ]
    runpy.run_module("dicom_utils.cli.dicom_types", run_name="__main__", alter_sys=True)
    captured = capsys.readouterr()
    assert "1 - None - ORIGINAL|PRIMARY|AXIAL" in captured.out
    assert len(captured.out.split("\n")) == 2

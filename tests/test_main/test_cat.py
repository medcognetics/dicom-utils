#!/usr/bin/env python
# -*- coding: utf-8 -*-
import runpy
import sys

import pytest


@pytest.fixture
def dicom_file():
    pydicom = pytest.importorskip("pydicom")
    return pydicom.data.get_testdata_file("CT_small.dcm")


def test_cat(dicom_file, capsys):
    sys.argv = [
        sys.argv[0],
        str(dicom_file),
    ]
    runpy.run_module("dicom_utils.cli.cat", run_name="__main__", alter_sys=True)
    captured = capsys.readouterr()
    assert "CT_small.dcm" in captured.out
    assert "1.3.6.1.4.1.5962.1.2.1.20040119072730.12322" in captured.out


def test_cat_tags(dicom_file, capsys):
    sys.argv = [sys.argv[0], str(dicom_file), "--tags", "StudyInstanceUID"]
    runpy.run_module("dicom_utils.cli.cat", run_name="__main__", alter_sys=True)
    captured = capsys.readouterr()
    assert "1.3.6.1.4.1.5962.1.2.1.20040119072730.12322" in captured.out
    assert "Study Date" not in captured.out

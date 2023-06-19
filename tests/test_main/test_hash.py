#!/usr/bin/env python
# -*- coding: utf-8 -*-
import runpy
import sys

import pytest


@pytest.fixture
def dicom_file():
    pydicom = pytest.importorskip("pydicom")
    return pydicom.data.get_testdata_file("CT_small.dcm")


@pytest.mark.parametrize("tag", ["StudyInstanceUID", "PixelData"])
@pytest.mark.parametrize("decompress", [False, True])
def test_hash(dicom_file, capsys, tag, decompress):
    sys.argv = [
        sys.argv[0],
        str(dicom_file),
        "--tag",
        tag,
    ]
    if decompress:
        sys.argv.append("--decompress")
    runpy.run_module("dicom_utils.cli.hash", run_name="__main__", alter_sys=True)
    captured = capsys.readouterr()
    assert captured.out.startswith("path,hash")
    assert "CT_small.dcm," in captured.out

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import runpy
import sys
from pathlib import Path

import pytest


@pytest.fixture
def dicom_file():
    pydicom = pytest.importorskip("pydicom")
    return pydicom.data.get_testdata_file("CT_small.dcm")


@pytest.mark.parametrize("out", [None, "foo.png"])
def test_dicom2img(dicom_file, tmp_path, out):
    if out is not None:
        path = Path(tmp_path, out)
        sys.argv = [sys.argv[0], str(dicom_file), "--noblock", "--output", str(path)]
    else:
        path = None
        sys.argv = [
            sys.argv[0],
            str(dicom_file),
            "--noblock",
        ]

    runpy.run_module("dicom_utils.cli.dicom2img", run_name="__main__", alter_sys=True)

    if out is not None:
        assert path.is_file()

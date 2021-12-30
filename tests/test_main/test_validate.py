#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import runpy
import shutil
import sys
from pathlib import Path

import pytest


@pytest.fixture
def dicom_file():
    pydicom = pytest.importorskip("pydicom")
    return pydicom.data.get_testdata_file("CT_small.dcm")


def test_validate(dicom_file, capsys, tmp_path):
    dest = Path(tmp_path, "file.dcm")
    shutil.copy(dicom_file, dest)
    sys.argv = [
        sys.argv[0],
        str(dest),
        "--no-color",
    ]
    with pytest.raises(SystemExit):
        runpy.run_module("dicom_utils.cli.validate", run_name="__main__", alter_sys=True)
    captured = capsys.readouterr()

    lines = [re.sub("[^A-Za-z0-9\t]+", "", s).split("\t") for s in captured.out.split("\n")]

    assert lines[0] == ["Tag", "Priority", "State", "Message"]
    assert len(lines) == 34
    # TODO consider validating the full output


@pytest.mark.parametrize("no_color", [True, False])
@pytest.mark.parametrize("failing_only", [True, False])
def test_validate_args(dicom_file, capsys, tmp_path, no_color, failing_only):
    dest = Path(tmp_path, "file.dcm")
    shutil.copy(dicom_file, dest)
    sys.argv = [
        sys.argv[0],
        str(dest),
    ]
    if no_color:
        sys.argv.append("--no-color")
    if failing_only:
        sys.argv.append("--failing-only")

    with pytest.raises(SystemExit):
        runpy.run_module("dicom_utils.cli.validate", run_name="__main__", alter_sys=True)
    captured = capsys.readouterr()
    assert captured.out

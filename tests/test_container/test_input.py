#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
from pathlib import Path

import pytest

from dicom_utils.container import RecordCollection
from dicom_utils.container.input import Input


@pytest.fixture
def dicom_files(tmp_path, dicom_file):
    paths = []
    for i in range(3):
        for j in range(3):
            dest = Path(tmp_path, f"subdir_{i}", f"file_{j}.dcm")
            dest.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(dicom_file, str(dest))
            paths.append(dest)
    return paths


class TestInput:
    def test_basic_input(self, tmp_path, dicom_files):
        source = tmp_path
        Path(tmp_path, "dest")
        p = list(Input(source, use_bar=False))
        assert len(p) == 1
        assert p[0][0] == "Case-1"
        assert isinstance(p[0][1], RecordCollection)

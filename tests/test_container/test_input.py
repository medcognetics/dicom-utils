#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
from pathlib import Path

import pytest

from dicom_utils.container.input import Input


@pytest.fixture
def suffix():
    return ".dcm"


@pytest.fixture
def dicom_files(tmp_path, dicom_file, suffix):
    paths = []
    for i in range(3):
        for j in range(3):
            dest = Path(tmp_path, f"subdir_{i}", f"file_{j}{suffix}")
            dest.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(dicom_file, str(dest))
            paths.append(dest)
    return paths


class TestInput:
    @pytest.mark.parametrize("suffix", [".dcm", "", ".123"])
    def test_basic_input(self, tmp_path, dicom_files):
        source = tmp_path
        Path(tmp_path, "dest")
        p = list(Input(source, use_bar=False))
        assert len(p) == 1
        key = p[0][0]
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert all(isinstance(k, str) for k in key)

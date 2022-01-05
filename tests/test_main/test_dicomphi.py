#!/usr/bin/env python
# -*- coding: utf-8 -*-
import runpy
import sys

from tests.test_main.test_dicom_types import dicom_folder


# Necessary so that "dicom_folder" is not seen as unused
dicom_folder = dicom_folder


def test_dicomphi(dicom_folder, tmp_path):
    sys.argv = [sys.argv[0], str(tmp_path)]
    runpy.run_module("dicom_utils.cli.dicomphi", run_name="__main__", alter_sys=True)

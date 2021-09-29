#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pydicom
import pytest
from pydicom.data import get_testdata_file


@pytest.fixture
def dicom_file():
    return get_testdata_file("CT_small.dcm")


@pytest.fixture
def dicom_object(dicom_file):
    with pydicom.dcmread(dicom_file) as dcm:
        yield dcm

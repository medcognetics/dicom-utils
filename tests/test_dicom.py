#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pydicom
import pytest

from dicom_utils import read_dicom_image


@pytest.fixture
def dicom_file():
    pydicom = pytest.importorskip("pydicom")
    return pydicom.data.get_testdata_file("CT_small.dcm")


@pytest.fixture
def dicom_object(dicom_file):
    with pydicom.dcmread(dicom_file) as dcm:
        yield dcm


class TestReadDicomImage:
    def test_shape(self, dicom_object):
        array = read_dicom_image(dicom_object)
        assert isinstance(array, np.ndarray)
        assert array.ndim == 3, "dims C x H x W"
        assert array.shape[0] == 1, "channel dim size == 1"
        assert array.shape[1] == 128, "height dim size == 128"
        assert array.shape[2] == 128, "width dim size == 128"

    def test_array_dtype(self, dicom_object):
        array = read_dicom_image(dicom_object)
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.int16

    def test_min_max_values(self, dicom_object):
        array = read_dicom_image(dicom_object)
        assert isinstance(array, np.ndarray)
        assert array.min() == 128, "min pixel value 128"
        assert array.max() == 2191, "max pixel value 2191"

    @pytest.mark.parametrize("shape_override", [None, (32, 32), (32, 32, 32)])
    def test_stop_before_pixels(self, dicom_object, shape_override):
        np.random.seed(42)
        array1 = read_dicom_image(dicom_object)
        array2 = read_dicom_image(dicom_object, stop_before_pixels=True, shape=shape_override)
        assert isinstance(array1, np.ndarray)
        assert isinstance(array2, np.ndarray)

        if shape_override is None:
            assert not (array1 == array2).all()  # type: ignore
            assert array1.shape == array2.shape  # type: ignore
        else:
            assert array2.shape == (1,) + shape_override  # type: ignore

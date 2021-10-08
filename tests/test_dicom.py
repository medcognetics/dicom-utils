#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pytest

from dicom_utils import KeepVolume, SliceAtLocation, UniformSample, read_dicom_image


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

    def test_invalid_TransferSyntaxUID_loose_interpretation(self, dicom_object):
        dicom_object.file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.4.90"  # Assign random invalid TransferSyntaxUID
        array = read_dicom_image(dicom_object)
        assert isinstance(array, np.ndarray)
        assert array.min() == 128, "min pixel value 128"
        assert array.max() == 2191, "max pixel value 2191"

    def test_invalid_TransferSyntaxUID_exception(self, dicom_object):
        dicom_object.file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.4.90"  # Assign random invalid TransferSyntaxUID
        with pytest.raises(ValueError) as e:
            read_dicom_image(dicom_object, strict_interp=True)
        assert "does not appear to be correct" in str(e), "The expected exception message was not returned."

    def test_invalid_PixelData(self, dicom_object):
        dicom_object.PixelData = b""
        with pytest.raises(ValueError) as e:
            read_dicom_image(dicom_object)
        expected_msg = "Unable to parse the pixel array after trying all possible TransferSyntaxUIDs."
        assert expected_msg in str(e), "The expected exception message was not returned."

    @pytest.mark.parametrize("shape_override", [None, (32, 32), (32, 32, 32)])
    def test_stop_before_pixels(self, dicom_object, shape_override):
        np.random.seed(42)
        array1 = read_dicom_image(dicom_object)
        array2 = read_dicom_image(dicom_object, stop_before_pixels=True, shape=shape_override)
        assert isinstance(array1, np.ndarray)
        assert isinstance(array2, np.ndarray)

        if shape_override is None:
            assert not (array1 == array2).all()
            assert array1.shape == array2.shape
        else:
            assert array2.shape == (1,) + shape_override

    @pytest.mark.parametrize(
        "handler",
        [
            KeepVolume(),
            SliceAtLocation(4),
            UniformSample(4, method="count"),
        ],
    )
    def test_volume_handling(self, dicom_object_3d, handler, mocker, transfer_syntax):
        spy = mocker.spy(handler, "__call__")
        F = 8
        dcm = dicom_object_3d(num_frames=F, syntax=transfer_syntax)
        array1 = read_dicom_image(dcm, volume_handler=spy, strict_interp=True)
        spy.assert_called_once()
        assert spy.mock_calls[0].args[0] == dcm, "handler should be called with DICOM object"
        assert array1.ndim < 4 or array1.shape[1] != 1, "3D dim should be squeezed when D=1"

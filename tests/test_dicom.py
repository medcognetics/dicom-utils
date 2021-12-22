#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pytest
from pydicom import DataElement

from dicom_utils import KeepVolume, SliceAtLocation, UniformSample, read_dicom_image
from dicom_utils.types import WINDOW_CENTER, WINDOW_WIDTH, Window


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

    @pytest.mark.parametrize(
        "apply,center,width",
        [
            pytest.param(True, None, None),
            pytest.param(True, DataElement(WINDOW_CENTER, "DS", 512), None),
            pytest.param(True, None, DataElement(WINDOW_WIDTH, "DS", 512)),
            pytest.param(True, DataElement(WINDOW_CENTER, "DS", 512), DataElement(WINDOW_WIDTH, "DS", 256)),
            pytest.param(True, DataElement(WINDOW_CENTER, "DS", 256), DataElement(WINDOW_WIDTH, "DS", 512)),
            pytest.param(
                True,
                DataElement(WINDOW_CENTER, "DS", [100, 200, 300]),
                DataElement(WINDOW_WIDTH, "DS", [200, 300, 400]),
            ),
            pytest.param(False, DataElement(WINDOW_CENTER, "DS", 512), DataElement(WINDOW_WIDTH, "DS", 256)),
            pytest.param(False, DataElement(WINDOW_CENTER, "DS", 256), DataElement(WINDOW_WIDTH, "DS", 512)),
            pytest.param(
                False,
                DataElement(WINDOW_CENTER, "DS", [100, 200, 300]),
                DataElement(WINDOW_WIDTH, "DS", [200, 300, 400]),
            ),
        ],
    )
    def test_apply_window(self, dicom_object, apply, center, width):
        # set metadata
        if center is not None:
            dicom_object[WINDOW_CENTER] = center
        if width is not None:
            dicom_object[WINDOW_WIDTH] = width

        window = Window.from_dicom(dicom_object)
        pixels = read_dicom_image(dicom_object, apply_window=False)
        window_pixels = read_dicom_image(dicom_object, apply_window=apply)

        if center is not None and width is not None and apply:
            assert (window_pixels >= 0).all()
            assert (window_pixels <= window.width).all()
            assert (window_pixels[pixels <= window.lower_bound] == 0).all()
            assert (window_pixels[pixels >= window.upper_bound] == window.upper_bound - window.lower_bound).all()

        elif apply:
            pixels = dicom_object.pixel_array
            assert window_pixels.min() == 0
            # tolerance of 1 here for rounding errors
            assert window_pixels.max() - (pixels.max() - pixels.min()) <= 1

        else:
            assert (window_pixels == pixels).all()

        window = Window.from_dicom(dicom_object)

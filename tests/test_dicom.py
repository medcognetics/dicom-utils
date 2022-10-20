#!/usr/bin/env python
# -*- coding: utf-8 -*-


import time

import numpy as np
import pydicom
import pytest
from numpy.random import default_rng
from pydicom.uid import ImplicitVRLittleEndian, RLELossless

from dicom_utils import KeepVolume, SliceAtLocation, UniformSample, read_dicom_image
from dicom_utils.dicom import data_handlers, default_data_handlers, is_inverted, set_pixels


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
        array2 = read_dicom_image(dicom_object, stop_before_pixels=True, override_shape=shape_override)
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

    def test_decoding_speed(self, dicom_file_j2k: str) -> None:
        # Make sure that our set of pixel data handlers is actually faster than the default set

        def time_decode() -> float:
            start_time = time.time()
            pydicom.dcmread(dicom_file_j2k).pixel_array
            return time.time() - start_time

        pydicom.config.pixel_data_handlers = default_data_handlers  # type: ignore
        default_decode_time = time_decode()

        pydicom.config.pixel_data_handlers = data_handlers  # type: ignore
        decode_time = time_decode()

        assert decode_time < default_decode_time, f"{decode_time} is not less than {default_decode_time}"

    def test_as_uint8(self, dicom_object):
        array = read_dicom_image(dicom_object, as_uint8=True)
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.uint8
        assert array.min() == 0
        assert array.max() == 255

    def test_read_rgb(self, dicom_object):
        test_file = pydicom.data.get_testdata_file("SC_rgb_rle.dcm")  # type: ignore
        dicom_object = pydicom.dcmread(test_file)
        array = read_dicom_image(dicom_object)
        assert isinstance(array, np.ndarray)
        assert array.shape[0] == 3
        assert array.shape[-2:] == (100, 100)
        assert array.dtype == np.uint8


def test_deprecated_is_inverted(dicom_object):
    with pytest.warns(DeprecationWarning):
        assert not is_inverted(dicom_object.PhotometricInterpretation)


@pytest.mark.parametrize(
    "rows,cols,num_frames,bits,orig_tsuid",
    [
        pytest.param(32, 32, 1, 16, ImplicitVRLittleEndian),
        pytest.param(32, 64, 1, 16, ImplicitVRLittleEndian),
        pytest.param(32, 64, 3, 16, ImplicitVRLittleEndian),
        pytest.param(32, 64, 3, 16, RLELossless),
    ],
)
def test_set_pixels(dicom_object, rows, cols, num_frames, bits, transfer_syntax, orig_tsuid):
    dicom_object.Rows = rows
    dicom_object.Columns = cols
    dicom_object.NumberOfFrames = num_frames
    dicom_object.file_meta.TransferSyntaxUID = orig_tsuid

    low = 0
    high = bits
    channels = 1 if dicom_object.PhotometricInterpretation.startswith("MONOCHROME") else 3
    size = tuple(x for x in (channels, num_frames, rows, cols) if x > 1)
    rng = default_rng(seed=42)
    arr = rng.integers(low, high, size, dtype=np.uint16)

    output = set_pixels(dicom_object, arr, transfer_syntax)
    arr_out = output.pixel_array
    assert isinstance(arr_out, np.ndarray)
    assert output.file_meta.TransferSyntaxUID == transfer_syntax

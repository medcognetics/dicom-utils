#!/usr/bin/env python
# -*- coding: utf-8 -*-


import time

import numpy as np
import pydicom
import pydicom.uid as puid
import pytest
from numpy.random import default_rng
from pydicom.pixels.decoders.base import get_decoder
from pydicom.uid import ImplicitVRLittleEndian, RLELossless

import dicom_utils
from dicom_utils import KeepVolume, ReduceVolume, SliceAtLocation, UniformSample, VolumeHandler, read_dicom_image
from dicom_utils.dicom import data_handlers, default_data_handlers, is_inverted, nvjpeg_decompress, set_pixels


@pytest.fixture
def pynvjpeg():
    return pytest.importorskip("pynvjpeg", reason="pynvjpeg is not installed")


class TestReadDicomImage:
    @pytest.mark.parametrize(
        "tsuid",
        [
            puid.ExplicitVRLittleEndian,
            puid.ImplicitVRLittleEndian,
            puid.JPEG2000Lossless,
            *puid.JPEGLSTransferSyntaxes,
        ],
    )
    def test_decoder(self, tsuid):
        assert get_decoder(tsuid) is not None

    def test_shape(self, dicom_object):
        array = read_dicom_image(dicom_object)
        assert isinstance(array, np.ndarray)
        assert array.ndim == 3, "dims C x H x W"
        assert array.shape[0] == 1, "channel dim size == 1"
        assert array.shape[1] == 128, "height dim size == 128"
        assert array.shape[2] == 128, "width dim size == 128"

    def test_array_dtype(self, dicom_object):
        array = read_dicom_image(dicom_object, rescale=False)
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.int16

    def test_min_max_values(self, dicom_object):
        array = read_dicom_image(dicom_object, rescale=False)
        assert isinstance(array, np.ndarray)
        assert array.min() == 128, "min pixel value 128"
        assert array.max() == 2191, "max pixel value 2191"

    def test_invalid_TransferSyntaxUID_loose_interpretation(self, dicom_object):
        dicom_object.file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.4.90"  # Assign random invalid TransferSyntaxUID
        array = read_dicom_image(dicom_object, use_nvjpeg=False, rescale=False)
        assert isinstance(array, np.ndarray)
        assert array.min() == 128, "min pixel value 128"
        assert array.max() == 2191, "max pixel value 2191"

    def test_invalid_TransferSyntaxUID_exception(self, dicom_object):
        dicom_object.file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.4.90"  # Assign random invalid TransferSyntaxUID
        with pytest.raises(ValueError) as e:
            read_dicom_image(dicom_object, strict_interp=True, use_nvjpeg=False)
        assert "does not appear to be correct" in str(e), "The expected exception message was not returned."

    def test_invalid_PixelData(self, dicom_object):
        dicom_object.PixelData = b""
        with pytest.raises(ValueError) as e:
            read_dicom_image(dicom_object, use_nvjpeg=False)
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
            pytest.param(lambda x: x, marks=pytest.mark.xfail(raises=TypeError)),
            KeepVolume(),
            SliceAtLocation(4),
            UniformSample(4, method="count"),
            ReduceVolume(output_frames=4),
        ],
    )
    def test_volume_handling(self, dicom_object_3d, handler, mocker, transfer_syntax):
        # NOTE: we need to patch handle_dicom for VolumeHandlers, and __call__ for other cases
        # There's an issue with mocking __call__ in all cases, possibly due to the use of a functools.partial
        if isinstance(handler, VolumeHandler):
            spy = mocker.patch.object(handler, "handle_dicom", wraps=handler.handle_dicom)
        else:
            spy = mocker.patch.object(handler, "__call__", wraps=handler)
        F = 8
        dcm = dicom_object_3d(num_frames=F, syntax=transfer_syntax)
        array1 = read_dicom_image(dcm, volume_handler=handler, strict_interp=True)
        spy.assert_called_once()
        assert spy.mock_calls[0].args[0] == dcm, "handler should be called with DICOM object"
        assert array1.ndim < 4 or array1.shape[1] != 1, "3D dim should be squeezed when D=1"

    @pytest.mark.parametrize("has_voi_lut", [True, False])
    def test_convert_3d_voi_lut(self, mocker, dicom_object_3d, has_voi_lut):
        dcm = dicom_object_3d(num_frames=8)
        if has_voi_lut:
            dcm.WindowCenter = 512
            dcm.WindowWidth = 512

        spy = mocker.spy(dicom_utils.dicom, "convert_frame_voi_lut")
        read_dicom_image(dcm, strict_interp=True)
        if has_voi_lut:
            spy.assert_not_called()
        else:
            spy.assert_called_once()

    def test_decoding_speed(self, dicom_file_j2k: str) -> None:
        # Make sure that our set of pixel data handlers is actually faster than the default set

        def time_decode() -> float:
            start_time = time.time()
            pydicom.dcmread(dicom_file_j2k).pixel_array
            return time.time() - start_time

        pydicom.config.pixel_data_handlers = default_data_handlers  # type: ignore
        default_decode_time = 2 * time_decode()

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

    @pytest.mark.parametrize(
        "dicom_fixture,use_nvjpeg,available,exp",
        [
            # 3D JPEG2000
            pytest.param("dicom_file_j2k_3d_uint16", True, True, True, marks=pytest.mark.usefixtures("pynvjpeg")),
            pytest.param("dicom_file_j2k_3d_uint16", None, True, True, marks=pytest.mark.usefixtures("pynvjpeg")),
            pytest.param("dicom_file_j2k_3d_uint16", True, False, True, id="cpu-fallback"),
            pytest.param("dicom_file_j2k_3d_uint16", None, False, False),
            pytest.param("dicom_file_j2k_3d_uint16", False, False, False),
            # 2D JPEG2000
            pytest.param("dicom_file_j2k_uint16", True, True, True, marks=pytest.mark.usefixtures("pynvjpeg")),
            pytest.param("dicom_file_j2k_uint16", None, True, True, marks=pytest.mark.usefixtures("pynvjpeg")),
            pytest.param("dicom_file_j2k_uint16", True, False, True, id="cpu-fallback"),
            pytest.param("dicom_file_j2k_uint16", None, False, False),
            pytest.param("dicom_file_j2k_uint16", False, False, False),
            # not JPEG2000, we should not attempt nvJPEG2K
            pytest.param("dicom_file_jpl14", True, True, False),
            pytest.param("dicom_file_jpl14", None, True, False),
            pytest.param("dicom_file_jpl14", True, False, False),
            pytest.param("dicom_file_jpl14", None, False, False),
            pytest.param("dicom_file_jpl14", False, False, False),
            # We should never run on int16 because it's not supported
            pytest.param("dicom_file_j2k_int16", True, True, False),
            pytest.param("dicom_file_j2k_int16", None, True, False),
            pytest.param("dicom_file_j2k_int16", True, False, False),
            pytest.param("dicom_file_j2k_int16", None, False, False),
            pytest.param("dicom_file_j2k_int16", False, False, False),
        ],
    )
    def test_nvjpeg_autoselect(self, request, mocker, dicom_fixture, use_nvjpeg, available, exp):
        dicom_file = request.getfixturevalue(dicom_fixture)

        # patch methods
        a = mocker.patch("dicom_utils.dicom.nvjpeg2k_is_available", return_value=available)

        # patch to use CPU instead of GPU
        def new(dcm, *args, **kwargs):
            if not a():
                raise ImportError("nvJPEG not available")
            return dcm.pixel_array

        m = mocker.patch("dicom_utils.dicom.nvjpeg_decompress", side_effect=new)

        # read and check
        with pydicom.dcmread(dicom_file) as dcm:
            read_dicom_image(dcm, use_nvjpeg=use_nvjpeg, nvjpeg_batch_size=1)
        assert m.called == exp

    @pytest.mark.ci_skip  # CircleCI will not have a GPU
    @pytest.mark.usefixtures("pynvjpeg")
    @pytest.mark.parametrize(
        "dicom_fixture",
        [
            "dicom_file_j2k_3d_uint16",
            "dicom_file_j2k_uint16",
            "dicom_file_jpl14",
            "dicom_file_j2k_int16",
        ],
    )
    def test_nvjpeg(self, request, dicom_fixture):
        dicom_file_j2k = request.getfixturevalue(dicom_fixture)

        ds = pydicom.dcmread(dicom_file_j2k)
        gpu_image = read_dicom_image(ds, use_nvjpeg=True, nvjpeg_batch_size=1)
        cpu_image = read_dicom_image(ds, use_nvjpeg=False)

        assert cpu_image.shape == gpu_image.shape
        assert (cpu_image == gpu_image).all()

    @pytest.mark.parametrize(
        "presentation,exp",
        [
            ("FOR PROCESSING", True),
            ("FOR PRESENTATION", True),
        ],
    )
    def test_for_algorithm_presentation_handling(self, mocker, dicom_object, presentation, exp):
        m1 = mocker.patch("dicom_utils.dicom.invert_color", return_value=dicom_object.pixel_array)
        dicom_object.PresentationIntentType = presentation
        dicom_object.PhotometricInterpretation = "MONOCHROME1"
        read_dicom_image(dicom_object)
        assert m1.called == exp, "invert_color not called as expected"

    def test_voi_lut_control(self, mocker, dicom_object):
        np.random.seed(42)
        spy = mocker.spy(dicom_utils.dicom, "apply_voi_lut")
        dicom_object.WindowCenter = 512
        dicom_object.WindowWidth = 512
        array1 = read_dicom_image(dicom_object, voi_lut=True)
        array2 = read_dicom_image(dicom_object, voi_lut=False)
        assert (array1 != array2).any()
        assert spy.call_count == 1

    def test_inversion_control(self, mocker, dicom_object):
        np.random.seed(42)
        spy = mocker.spy(dicom_utils.dicom, "invert_color")
        dicom_object.PhotometricInterpretation = "MONOCHROME1"
        array1 = read_dicom_image(dicom_object, inversion=True)
        array2 = read_dicom_image(dicom_object, inversion=False)
        assert (array1 != array2).any()
        assert spy.call_count == 1

    def test_rescale_control(self, mocker, dicom_object):
        np.random.seed(42)
        spy = mocker.spy(dicom_utils.dicom, "apply_rescale")
        dicom_object.RescaleIntercept = 1
        dicom_object.RescaleSlope = 2
        array1 = read_dicom_image(dicom_object, rescale=True)
        array2 = read_dicom_image(dicom_object, rescale=False)
        assert (array1 != array2).any()
        assert spy.call_count == 1


@pytest.mark.ci_skip  # CircleCI will not have a GPU
@pytest.mark.usefixtures("pynvjpeg")
@pytest.mark.parametrize(
    "dicom_fixture",
    [
        "dicom_file_j2k_3d_uint16",
        "dicom_file_j2k_uint16",
        pytest.param("dicom_file_jpl14", marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param("dicom_file_j2k_int16", marks=pytest.mark.xfail(raises=ValueError)),
    ],
)
def test_nvjpeg_decompress(request, dicom_fixture):
    dicom_file_j2k = request.getfixturevalue(dicom_fixture)

    ds = pydicom.dcmread(dicom_file_j2k)
    gpu_image = nvjpeg_decompress(ds, batch_size=1)
    ds = pydicom.dcmread(dicom_file_j2k)
    cpu_image = ds.pixel_array

    assert cpu_image.shape == gpu_image.shape
    assert (cpu_image == gpu_image).all()


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

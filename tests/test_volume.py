#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

import dicom_utils
from dicom_utils import KeepVolume, RandomSlice, ReduceVolume, SliceAtLocation, UniformSample


class TestKeepVolume:
    def test_array(self):
        x = np.random.rand(10, 10)
        sampler = KeepVolume()
        result = sampler(x)
        assert (x == result).all()

    def test_dicom(self, dicom_object_3d, transfer_syntax):
        N = 8
        dcm = dicom_object_3d(N, syntax=transfer_syntax)
        sampler = KeepVolume()
        result = sampler(dcm)
        assert type(result) == type(dcm)
        assert result.NumberOfFrames == dcm.NumberOfFrames
        assert (result.pixel_array == dcm.pixel_array).all()


class TestSliceAtLocation:
    @pytest.mark.parametrize(
        "center,before,after,stride,index",
        [
            pytest.param(5, 0, 0, 1, lambda a: a[5]),
            pytest.param(5, 1, 0, 1, lambda a: a[4:6]),
            pytest.param(5, 1, 1, 1, lambda a: a[4:7]),
            pytest.param(5, 0, 0, 2, lambda a: a[5]),
            pytest.param(5, 2, 2, 2, lambda a: a[3:8:2]),
        ],
    )
    def test_array(self, center, before, after, stride, index):
        x = np.random.rand(10, 10)
        sampler = SliceAtLocation(center, before, after, stride)
        result = sampler(x)
        assert type(x) == type(result)
        assert (index(x) == result).all()

    @pytest.mark.parametrize(
        "center,before,after,stride",
        [
            pytest.param(5, 0, 0, 1),
            pytest.param(5, 1, 0, 1),
            pytest.param(5, 1, 1, 1),
            pytest.param(5, 0, 0, 2),
            pytest.param(5, 2, 2, 2),
        ],
    )
    def test_dicom(self, dicom_object_3d, center, before, after, stride, transfer_syntax):
        N = 8
        dcm = dicom_object_3d(N, syntax=transfer_syntax)

        sampler = SliceAtLocation(center, before, after, stride)
        result = sampler(dcm)
        assert dcm.NumberOfFrames == N, "the input dicom object was modified"
        assert type(result) == type(dcm)
        assert (sampler(dcm.pixel_array) == result.pixel_array).all()


class TestUniformSample:
    @pytest.mark.parametrize(
        "size,amount,method,index",
        [
            pytest.param(10, 2, "stride", lambda a: a[::2]),
            pytest.param(12, 2, "stride", lambda a: a[::2]),
            pytest.param(10, 4, "count", lambda a: a[:: 10 // 4]),
            pytest.param(12, 4, "count", lambda a: a[:: 12 // 4]),
            pytest.param(2, 4, "count", lambda a: a),
            pytest.param(2, 1, "stride", lambda a: a),
        ],
    )
    def test_array(self, size, amount, method, index):
        x = np.random.rand(size, 10)
        sampler = UniformSample(amount, method)
        result = sampler(x)
        assert type(x) == type(result)
        expected = index(x)
        try:
            assert (expected == result).all()
        except Exception:
            assert expected == result

    @pytest.mark.parametrize(
        "amount,method",
        [
            pytest.param(2, "stride"),
            pytest.param(2, "stride"),
            pytest.param(4, "count"),
            pytest.param(4, "count"),
            pytest.param(4, "count"),
            pytest.param(1, "stride"),
        ],
    )
    def test_dicom(self, dicom_object_3d, amount, method, transfer_syntax):
        N = 8
        dcm = dicom_object_3d(N, syntax=transfer_syntax)

        sampler = UniformSample(amount, method)
        result = sampler(dcm)
        assert dcm.NumberOfFrames == N, "the input dicom object was modified"
        assert type(result) == type(dcm)
        assert (sampler(dcm.pixel_array) == result.pixel_array).all()


class TestReduceVolume:
    def test_dicom(self, dicom_object_3d, transfer_syntax):
        N = 8
        dcm = dicom_object_3d(N, syntax=transfer_syntax)
        sampler = ReduceVolume()
        expected = np.max(dcm.pixel_array, axis=0)
        result = sampler(dcm)
        assert type(result) == type(dcm)
        assert result.NumberOfFrames == 1
        assert (result.pixel_array == expected).all()

    @pytest.mark.parametrize("output_frames", [1, 2, 4, 8])
    def test_multi_frame(self, dicom_object_3d, output_frames):
        N = 8
        dcm = dicom_object_3d(N)
        sampler = ReduceVolume(output_frames=output_frames)
        expected = dcm.pixel_array.reshape(output_frames, N // output_frames, *dcm.pixel_array.shape[1:]).max(axis=1)
        result = sampler(dcm)
        assert type(result) == type(dcm)
        assert (result.pixel_array == expected).all()
        assert result.NumberOfFrames == output_frames

    def test_multi_frame_noop(self, dicom_object_3d):
        N = 8
        dcm = dicom_object_3d(N)
        sampler = ReduceVolume(output_frames=N)
        result = sampler(dcm)
        assert type(result) == type(dcm)
        assert result.NumberOfFrames == N
        assert (result.pixel_array == dcm.pixel_array).all()

    @pytest.mark.parametrize(
        "skip,output_frames",
        [
            (0, 1),
            (2, 3),
            # In this case we widen the range of the output frames to include the edge frames
            (4, 3),
            # In this case we can't satisfy 9 output frames from 8 total frames
            pytest.param(4, 9, marks=pytest.mark.xfail(raises=RuntimeError, strict=True)),
        ],
    )
    def test_skip_edge_frames(self, dicom_object_3d, skip, output_frames):
        N = 8
        dcm = dicom_object_3d(N)
        sampler = ReduceVolume(output_frames=output_frames, skip_edge_frames=skip)
        result = sampler(dcm)
        assert type(result) == type(dcm)
        assert result.NumberOfFrames == output_frames

    def test_skip_edge_frames_noop_reduction(self, dicom_object_3d):
        N = 8
        dcm = dicom_object_3d(N)
        sampler = ReduceVolume(output_frames=N - 4, skip_edge_frames=2)
        expected = dcm.pixel_array[2:-2]
        result = sampler(dcm)
        assert type(result) == type(dcm)
        assert result.NumberOfFrames == N - 4
        assert (result.pixel_array == expected).all()

    @pytest.mark.parametrize("use_nvjpeg", [False, None])
    def test_decompress(self, mocker, use_nvjpeg, dicom_object_3d, transfer_syntax):
        spy = mocker.spy(dicom_utils.dicom, "decompress")
        N = 8
        dcm = dicom_object_3d(N, syntax=transfer_syntax)
        sampler = ReduceVolume()
        sampler(dcm, use_nvjpeg=use_nvjpeg)
        spy.assert_called()
        for call in spy.mock_calls:
            assert call.kwargs["use_nvjpeg"] == use_nvjpeg


class TestRandomSlice:
    def test_array(self):
        x = np.random.rand(10, 10)
        sampler = RandomSlice()
        result = sampler(x)
        assert result.shape == (1, 10)

    def test_dicom(self, dicom_object_3d, transfer_syntax):
        N = 8
        dcm = dicom_object_3d(N, syntax=transfer_syntax)
        sampler = RandomSlice()
        result = sampler(dcm)
        assert type(result) == type(dcm)
        assert result.NumberOfFrames == 1

    def test_seed(self):
        x = np.random.rand(10, 10)
        sampler1 = RandomSlice(seed=42)
        sampler2 = RandomSlice(seed=42)
        sampler3 = RandomSlice(seed=0)
        result1 = sampler1(x)
        result2 = sampler2(x)
        result3 = sampler3(x)
        assert (result1 == result2).all()
        assert not (result1 == result3).all()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from dicom_utils import KeepVolume, SliceAtLocation, UniformSample


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

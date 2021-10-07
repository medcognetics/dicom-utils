#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from dicom_utils import KeepVolume, SliceAtLocation, UniformSample


def test_keep_volume():
    x = np.random.rand(10, 10)
    sampler = KeepVolume()
    result = sampler(x)
    assert (x == result).all()


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
def test_slice_at_location(center, before, after, stride, index):
    x = np.random.rand(10, 10)
    sampler = SliceAtLocation(center, before, after, stride)
    result = sampler(x)
    assert type(x) == type(result)
    assert (index(x) == result).all()


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
def test_uniform_sample(size, amount, method, index):
    x = np.random.rand(size, 10)
    sampler = UniformSample(amount, method)
    result = sampler(x)
    assert type(x) == type(result)
    expected = index(x)
    try:
        assert (expected == result).all()
    except Exception:
        assert expected == result

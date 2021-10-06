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
    "size,count,stride,index",
    [
        pytest.param(10, None, None, None, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(10, None, 2, lambda a: a[::2]),
        pytest.param(12, None, 2, lambda a: a[::2]),
        pytest.param(10, 4, None, lambda a: a[:: 10 // 4]),
        pytest.param(12, 4, None, lambda a: a[:: 12 // 4]),
        pytest.param(2, 4, None, lambda a: a),
        pytest.param(2, None, 1, lambda a: a),
    ],
)
def test_uniform_sample(size, count, stride, index):
    x = np.random.rand(size, 10)
    sampler = UniformSample(stride, count)
    result = sampler(x)
    assert type(x) == type(result)
    expected = index(x)
    try:
        assert (expected == result).all()
    except Exception:
        assert expected == result

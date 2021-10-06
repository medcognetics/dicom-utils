#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Optional, Sized, SupportsIndex, TypeVar


T = TypeVar("T", bound=SupportsIndex)


class VolumeHandler(ABC):
    r"""Base class for classes that manipulate 3D Volumes"""

    @abstractmethod
    def __call__(self, x: T) -> T:
        ...


class KeepVolume(VolumeHandler):
    r"""Retains the entire input volume"""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __call__(self, x: T) -> T:
        return x


class SliceAtLocation(VolumeHandler):
    r"""Samples the input volume at centered on a given slice with optional context frames.

    Args:
        center:
            The slice about which to sample

        before:
            Optional frames to sample before ``center``.

        after:
            Optional frames to sample after ``center``.

        stride:
            If given, the stride between sampled frames
    """

    def __init__(
        self,
        center: int,
        before: int = 0,
        after: int = 0,
        stride: int = 1,
    ):
        self.center = center
        self.stride = stride
        self.before = before
        self.after = after

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}("
        s += f"center={self.center}, "
        s += f"before={self.before}, "
        s += f"stride={self.stride}"
        s += ")"
        return s

    def __call__(self, x: T) -> T:
        start = self.center - self.before
        end = self.center + self.after + 1
        result = x[start : end : self.stride]  # type: ignore
        return result


class UniformSample(VolumeHandler):
    r"""Samples the input volume at centered on a given slice with optional context frames.
    Either ``stride`` or ``count`` must be provided.

    Args:
        stride:
            If given, the stride between sampled frames

        count:
            If given, the total number of frames to include in the output. Stride will be
            computed as ``min(num_slices // count, 1)``.
    """

    def __init__(
        self,
        stride: Optional[int] = None,
        count: Optional[int] = None,
    ):
        if stride is None and count is None:
            raise ValueError("Either `stride` or `count` must be provided")
        self.stride = stride
        self.count = count

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}("
        s += f"count={self.count}, "
        s += f"stride={self.stride}"
        s += ")"
        return s

    def __call__(self, x: T) -> T:
        if self.stride:
            result = x[:: self.stride]  # type: ignore
        elif self.count:
            assert isinstance(x, Sized)
            total_slices = len(x)
            stride = max(total_slices // self.count, 1)
            result = x[::stride]  # type: ignore
        else:
            raise ValueError(f"Either `stride` or `count` are required: {self.stride}, {self.count}")
        return result

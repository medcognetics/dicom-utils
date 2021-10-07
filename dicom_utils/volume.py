#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Sized, SupportsIndex, TypeVar


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
        amount:
            When when ``method='count'``, the number of frames to sample.
            When ``method='stride'``, the stride between sampled frames.

        method:
            Either ``'count'`` or ``'stride'``.
    """

    def __init__(
        self,
        amount: int,
        method: str = "count",
    ):
        if method not in ("count", "stride"):
            raise ValueError(f"`method` {method} must be one of 'count', 'stride'")
        self.amount = amount
        self.method = method

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}("
        s += f"amount={self.amount}, "
        s += f"method={self.method}"
        s += ")"
        return s

    def __call__(self, x: T) -> T:
        if self.method == "stride":
            result = x[:: self.amount]  # type: ignore
        elif self.method == "count":
            assert isinstance(x, Sized)
            total_slices = len(x)
            stride = max(total_slices // self.amount, 1)
            result = x[::stride]  # type: ignore
        else:
            raise ValueError(f"Invalid method {self.method}")
        return result

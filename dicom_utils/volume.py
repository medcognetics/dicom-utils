#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import islice
from typing import Any, Iterator, Optional, Protocol, Sized, SupportsInt, Tuple, TypeVar, Union, overload

import numpy as np
from pydicom import Dataset
from pydicom.encaps import encapsulate, generate_pixel_data_frame


class SupportsGetItem(Protocol):
    def __getitem__(self, key: Any) -> Any:
        ...


T = TypeVar("T", bound=SupportsGetItem)
U = TypeVar("U", bound=Dataset)


class VolumeHandler(ABC):
    r"""Base class for classes that manipulate 3D Volumes"""

    @abstractmethod
    def get_indices(self, total_frames: Optional[int]) -> Tuple[int, int, int]:
        ...

    @overload
    def __call__(self, x: T) -> T:
        ...

    @overload
    def __call__(self, x: U) -> U:
        ...

    def __call__(self, x: Union[T, U]) -> Union[T, U]:
        if isinstance(x, Dataset):
            return self.slice_dicom(x)
        else:
            return self.slice_array(x)

    def slice_array(self, x: T) -> T:
        r"""Slices an array input according to :func:`get_indices`"""
        num_frames = len(x) if isinstance(x, Sized) else None
        start, stop, stride = self.get_indices(num_frames)
        result = x[slice(start, stop, stride)]
        return result

    def slice_dicom(self, dcm: U) -> U:
        r"""Slices a DICOM object input according to :func:`get_indices`.

        .. note:
            Unlike :func:`slice_array`, this function can perform slicing on compressed DICOMs
            with out needing to decompress all frames. This can provide a substantial performance gain.

        """
        # copy dicom and read key tags
        dcm = deepcopy(dcm)
        num_frames: Optional[SupportsInt] = dcm.get("NumberOfFrames", None)
        num_frames = int(num_frames) if num_frames is not None else None
        is_compressed: bool = dcm.file_meta.TransferSyntaxUID.is_compressed

        start, stop, stride = self.get_indices(num_frames)

        # read data
        if is_compressed:
            frame_iterator: Iterator = generate_pixel_data_frame(dcm.PixelData, num_frames)
            frames = list(islice(frame_iterator, start, stop, stride))
            new_pixel_data = encapsulate(frames)
        else:
            all_frames: np.ndarray = dcm.pixel_array
            frames = all_frames[start:stop:stride]
            new_pixel_data = frames.tobytes()

        out_frames = len(frames)
        dcm.NumberOfFrames = out_frames
        dcm.PixelData = new_pixel_data
        return dcm


class KeepVolume(VolumeHandler):
    r"""Retains the entire input volume"""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def get_indices(self, total_frames: Optional[int]) -> Tuple[int, Optional[int], int]:
        return 0, None, 1


class SliceAtLocation(VolumeHandler):
    r"""Samples the input volume at centered on a given slice with optional context frames.

    Args:
        center:
            The slice about which to sample. Defaults to num_frames / 2

        before:
            Optional frames to sample before ``center``.

        after:
            Optional frames to sample after ``center``.

        stride:
            If given, the stride between sampled frames
    """

    def __init__(
        self,
        center: Optional[int] = None,
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

    def slice_dicom(self, dcm: U) -> U:
        num_frames: Optional[SupportsInt] = dcm.get("NumberOfFrames", None)
        if num_frames is None and self.center is None:
            raise AttributeError(f"`NumberOfFrames` cannot be absent when `{self.__class__.__name__}.center` is `None`")
        return super().slice_dicom(dcm)

    def get_indices(self, total_frames: Optional[int]) -> Tuple[int, int, int]:
        if self.center is None and total_frames is None:
            raise ValueError(f"`total_frames` cannot be `None` when `{self.__class__.__name__}.center` is `None`")
        center = self.center if self.center is not None else total_frames // 2
        start = center - self.before
        end = center + self.after + 1
        return start, end, self.stride


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

    def get_indices(self, total_frames: Optional[int]) -> Tuple[int, Optional[int], int]:
        if self.method == "stride":
            return 0, None, self.amount
        elif self.method == "count":
            assert total_frames is not None
            stride = max(total_frames // self.amount, 1)
            return 0, None, stride
        else:
            raise ValueError(f"Invalid method {self.method}")

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import islice
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Optional,
    Protocol,
    Sized,
    SupportsInt,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
from pydicom import Dataset
from pydicom.encaps import encapsulate, generate_pixel_data_frame
from pydicom.pixel_data_handlers import numpy_handler
from pydicom.pixel_data_handlers.util import reshape_pixel_array
from pydicom.uid import ImplicitVRLittleEndian


class SupportsGetItem(Protocol):
    def __getitem__(self, key: Any) -> Any:
        ...


T = TypeVar("T", bound=SupportsGetItem)
U = TypeVar("U", bound=Dataset)


def dicom_copy(dcm: U) -> U:
    # Avoid multiple copies of PixelData which can be 100s of MB
    pixel_data = dcm.PixelData
    del dcm.PixelData
    if hasattr(dcm, "_pixel_array"):
        del dcm._pixel_array  # Delete possibly cached interpretation of PixelData
    new_dcm = deepcopy(dcm)
    dcm.PixelData = new_dcm.PixelData = pixel_data
    return new_dcm


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
            return self.handle_dicom(x)
        else:
            return self.handle_array(x)

    def handle_dicom(self, dcm: U) -> U:
        num_frames: Optional[SupportsInt] = dcm.get("NumberOfFrames", None)
        num_frames = int(num_frames) if num_frames is not None else None
        start, stop, stride = self.get_indices(num_frames)
        return self.slice_dicom(dcm, start, stop, stride)

    def handle_array(self, x: T) -> T:
        num_frames = len(x) if isinstance(x, Sized) else None
        start, stop, stride = self.get_indices(num_frames)
        return cast(T, self.slice_array(cast(SupportsGetItem, x), start, stop, stride))

    @classmethod
    def iterate_frames(cls, dcm: Dataset) -> Iterator[bytes]:
        is_compressed: bool = dcm.file_meta.TransferSyntaxUID.is_compressed
        num_frames: Optional[SupportsInt] = dcm.get("NumberOfFrames", None)
        num_frames = int(num_frames) if num_frames is not None else None
        if is_compressed:
            for frame in generate_pixel_data_frame(dcm.PixelData, num_frames):
                yield frame
        else:
            # manually call the numpy handler so we can read the full array as read only.
            # this avoid memory duplication
            arr = numpy_handler.get_pixeldata(dcm, read_only=True)
            arr = reshape_pixel_array(dcm, arr)
            for frame in arr:
                yield frame.tobytes()

    @classmethod
    def slice_array(cls, x: T, start: int, stop: int, stride: int) -> T:
        r"""Slices an array input according to :func:`get_indices`"""
        len(x) if isinstance(x, Sized) else None
        result = cast(SupportsGetItem, x)[slice(start, stop, stride)]
        return result

    @classmethod
    def update_pixel_data(cls, dcm: U, frames: List[bytes], preserve_compression: bool = True) -> U:
        r"""Updates PixelData with a new sequence of frames, accounting for compression type"""
        is_compressed: bool = dcm.file_meta.TransferSyntaxUID.is_compressed
        if is_compressed and preserve_compression:
            new_pixel_data = encapsulate(frames)
        else:
            new_pixel_data = b"".join(frames)
            # if dcm was compressed, we cant guarantee that the libraries needed to compress new
            # frames to that transfer syntax will be available. to account for this, just change
            # the old TSUID to an uncompressed variant and attach frames without compression
            if is_compressed:
                dcm.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        dcm.PixelData = new_pixel_data
        dcm.NumberOfFrames = len(frames)
        return dcm

    @classmethod
    def slice_dicom(cls, dcm: U, start: int, stop: int, stride: int) -> U:
        r"""Slices a DICOM object input according to :func:`get_indices`.

        .. note:
            Unlike :func:`slice_array`, this function can perform slicing on compressed DICOMs
            with out needing to decompress all frames. This can provide a substantial performance gain.

        """
        # copy dicom and read key tags
        dcm = dicom_copy(dcm)
        num_frames: Optional[SupportsInt] = dcm.get("NumberOfFrames", None)
        num_frames = int(num_frames) if num_frames is not None else None
        is_compressed: bool = dcm.file_meta.TransferSyntaxUID.is_compressed

        # read sliced frames
        frame_iterator = cls.iterate_frames(dcm)
        frames = list(islice(frame_iterator, start, stop, stride))
        if not frames:
            raise IndexError("No frames remain in the sliced DICOM")
        if not all(frames):
            raise IndexError("One or more frames had no contents")

        # update dicom object and return
        dcm = cls.update_pixel_data(dcm, frames)
        assert dcm.NumberOfFrames == len(frames)
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

    def handle_dicom(self, dcm: U) -> U:
        num_frames: Optional[SupportsInt] = dcm.get("NumberOfFrames", None)
        if num_frames is None and self.center is None:
            raise AttributeError(f"`NumberOfFrames` cannot be absent when `{self.__class__.__name__}.center` is `None`")
        return super().handle_dicom(dcm)

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


class ReduceVolume(VolumeHandler):
    r"""Base class for classes that manipulate 3D Volumes"""

    def __init__(
        self,
        reduction: Callable[..., np.ndarray] = np.maximum,
    ):
        self.reduction = reduction

    def get_indices(self, total_frames: Optional[int]) -> Tuple[int, int, int]:
        raise NotImplementedError

    def handle_dicom(self, dcm: U) -> U:
        # slice dicom at index 0 and read the associated numpy array
        sliced = self.slice_dicom(dcm, 0, 1, 1)
        arr = sliced.pixel_array

        # iterate through remaining slices, applying reduction in-place
        for i in range(1, dcm.NumberOfFrames):
            other = self.slice_dicom(dcm, i, i + 1, 1).pixel_array
            arr = self.reduction(arr, other, out=arr)
        dcm = self.update_pixel_data(sliced, [arr.tobytes()], preserve_compression=False)
        dcm.pixel_array
        return dcm

    def handle_array(self, x: T) -> T:
        raise NotImplementedError

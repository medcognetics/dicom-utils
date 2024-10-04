#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
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
from pydicom.encaps import encapsulate, generate_frames
from pydicom.pixel_data_handlers import numpy_handler
from pydicom.pixels.utils import reshape_pixel_array
from pydicom.uid import ImplicitVRLittleEndian
from registry import Registry


class SupportsGetItem(Protocol):
    def __getitem__(self, key: Any) -> Any: ...


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


VOLUME_HANDLERS = Registry("volume handlers")


class VolumeHandler(ABC):
    r"""Base class for classes that manipulate 3D Volumes"""

    @abstractmethod
    def get_indices(self, total_frames: Optional[int]) -> Tuple[int, int, int]: ...

    @overload
    def __call__(self, x: T) -> T: ...

    @overload
    def __call__(self, x: U) -> U: ...

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
            for frame in generate_frames(dcm.PixelData, number_of_frames=num_frames):
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
        center = self.center if self.center is not None else cast(int, total_frames) // 2
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
    r"""Reduces the input volume to a subset of frames by applying a reduction function. This volume handler
    will need to decompress frames in order to apply the reduction function. To support this, NVJPEG parameters
    are accepted by :func:`__call__` and passed to :func:`dicom_utils.dicom.decompress`.

    Args:
        reduction:
            A function that takes two frames and an ``out`` argument and returns the reduced frame.
            The ``out`` argument is used to avoid allocating new memory for the reduced frame.

        output_frames:
            The number of frames to output. Defaults to ``1``, meaning the entire volume is reduced to a single frame.

        skip_edge_frames:
            The number of frames to skip at the beginning and end of the volume. This is useful for removing
            artifacts from the beginning and end of the volume.

    Raises:
        RuntimeError: If no frames remain in the sliced DICOM
    """

    def __init__(
        self,
        reduction: Callable[..., np.ndarray] = cast(Any, np.maximum),
        output_frames: int = 1,
        skip_edge_frames: int = 0,
    ):
        self.reduction = reduction
        self.output_frames = output_frames
        self.skip_edge_frames = skip_edge_frames

    def __call__(self, x: Union[T, U], use_nvjpeg: Optional[bool] = None, **kwargs) -> Union[T, U]:
        if isinstance(x, Dataset):
            return self.handle_dicom(x, use_nvjpeg=use_nvjpeg, **kwargs)
        else:
            return self.handle_array(x)

    def get_indices(self, total_frames: Optional[int]) -> Tuple[int, int, int]:
        raise NotImplementedError

    def handle_dicom(self, dcm: U, use_nvjpeg: Optional[bool] = None, **kwargs) -> U:
        chunks = [chunk.tobytes() for chunk in self.iterate_chunks(dcm, self.reduction, use_nvjpeg, **kwargs)]
        if not chunks:
            raise RuntimeError("No frames remain in the sliced DICOM")
        assert len(chunks) == self.output_frames

        dcm = self.update_pixel_data(dcm, chunks, preserve_compression=False)
        assert not dcm.file_meta.TransferSyntaxUID.is_compressed, "DICOM should be decompressed after ReduceVolume"
        return dcm

    def handle_array(self, x: T) -> T:
        raise NotImplementedError

    def iterate_chunks(
        self,
        dcm: Dataset,
        reduction: Optional[Callable[..., np.ndarray]],
        use_nvjpeg: Optional[bool] = None,
        **kwargs,
    ) -> Iterator[np.ndarray]:
        r"""Iterates through the DICOM volume, yielding chunks based on the reduction parameters.
        If a reduction is given, each chunk will be reduced before being yielded.
        """
        # TODO: This is a bit of a hack to avoid circular imports. Fixing it will require a larger refactor.
        from .dicom import decompress

        start = self.skip_edge_frames
        stop = dcm.NumberOfFrames - self.skip_edge_frames

        # If the chunk size is less than 1, we need to adjust the start and stop indices.
        # We do this by widening the range between start and stop until we have a chunk size of 1
        # or until we reach the end of the volume. Since we are widening symetrically, we clip
        # the start index to a minimum of 0 so every possible frame is tried.
        while (chunk_size := (stop - start) // self.output_frames) < 1:
            start = max(start - 1, 0)
            stop += 1
            if stop > dcm.NumberOfFrames:
                raise RuntimeError("Could not find a valid chunk size")
        if chunk_size < 1:
            # TODO: consider returning the original volume + duplicating frames to fill the output
            raise RuntimeError("Could not find a valid chunk size")

        for yield_count in range(self.output_frames):
            # If this is the last chunk, make sure it includes all remaining frames
            is_last_chunk = yield_count == self.output_frames - 1
            chunk_end = stop if is_last_chunk else start + chunk_size

            # Slice the chunk
            assert start < chunk_end <= stop, f"Invalid chunk start/end: {start}, {chunk_end}, {stop}"
            sliced = self.slice_dicom(dcm, start, chunk_end, stride=1)

            # If no reduction is given, just yield the decompressed chunk
            if reduction is None or chunk_size == 1:
                sliced = decompress(sliced, use_nvjpeg=use_nvjpeg, **kwargs)
                yield sliced.pixel_array

            # Iterate through the chunk, applying the reduction in-place.
            # We never want to store more of the volume than is needed to conserve memory.
            else:
                arr = self.slice_dicom(sliced, 0, 1, stride=1).pixel_array
                for i in range(1, sliced.NumberOfFrames):
                    # TODO: it is less computationally efficient to decompress each frame individually,
                    # but it is more memory efficient. Memory efficiency is more important given the limitations
                    # of the deployment environment. Consider a better solution.
                    other = self.slice_dicom(sliced, i, i + 1, 1)
                    other = decompress(other, use_nvjpeg=use_nvjpeg, **kwargs)
                    arr = reduction(arr, other.pixel_array, out=arr)
                yield arr

            start += chunk_size


class RandomSlice(VolumeHandler):
    r"""Samples the input volume at a random slice

    Args:
        seed: The random seed to use
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def get_indices(self, total_frames: Optional[int]) -> Tuple[int, int, int]:
        if total_frames is None:
            raise ValueError("`total_frames` cannot be `None`")
        index = self.rng.randint(0, total_frames - 1)
        return index, index + 1, 1


# Register some default handlers
VOLUME_HANDLERS(name="keep")(KeepVolume)
VOLUME_HANDLERS(name="max")(ReduceVolume)
VOLUME_HANDLERS(name="mean", reduction=np.mean)(ReduceVolume)
VOLUME_HANDLERS(name="slice")(SliceAtLocation)

# Multi-frame reductions
for output_frames in (1, 8, 10, 16):
    for skip_edge_frames in (0, 5, 10):
        VOLUME_HANDLERS(
            name=f"max-{output_frames}-{skip_edge_frames}",
            output_frames=output_frames,
            skip_edge_frames=skip_edge_frames,
        )(ReduceVolume)

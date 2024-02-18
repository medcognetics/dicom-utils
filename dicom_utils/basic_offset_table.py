#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import copy
from functools import wraps
from inspect import signature
from struct import pack
from typing import Any, Callable, ClassVar, Final, Iterator, Sequence, TypeVar, Union, cast

from pydicom.filebase import DicomBytesIO

from .types import Dicom


StreamType = Union[bytes, DicomBytesIO, Dicom]
T = TypeVar("T")

START: Final = 0
PREFIX: Final = 0
BYTES_PER_LONG: Final = 4
LENGTH: Final = 4
OFFSETS: Final = 8


def _prepare_stream(stream: StreamType) -> DicomBytesIO:
    if isinstance(stream, Dicom):
        if not hasattr(stream, "PixelData"):
            raise ValueError("Input DICOM object does not have PixelData")
        stream = DicomBytesIO(stream.PixelData)
    elif isinstance(stream, bytes):
        stream = DicomBytesIO(stream)
    elif not isinstance(stream, DicomBytesIO):
        raise TypeError(type(stream))
    stream.is_little_endian = True
    stream.seek(0)
    return stream


# TODO: annotated with ParamSpec once Python 3.10 is well adopted
def prepare_stream(f: Callable[..., T]) -> Callable[..., T]:
    r"""Decorator to prepare args annotated as DicomBytesIO from a StreamType.
    Automatically converts the byte stream and seeks to the start of stream.
    """
    param_dict = signature(f).parameters
    param_list = list(param_dict)

    @wraps(f)
    def wrapper(*args, **kwargs) -> T:
        args = tuple(
            a if param_dict[param_list[i]].annotation != DicomBytesIO else _prepare_stream(a)
            for i, a in enumerate(args)
        )
        kwargs = {k: v if param_dict[k].annotation != DicomBytesIO else _prepare_stream(v) for k, v in kwargs.items()}
        return f(*args, **kwargs)

    return wrapper


def restore_pointer(f: Callable[..., T]) -> Callable[..., T]:
    r"""Decorator to restore the stream position after a function call"""

    @wraps(f)
    def wrapper(self: "BasicOffsetTable", *args, **kwargs) -> T:
        pos = self.fp.tell()
        result = f(self, *args, **kwargs)
        self.fp.seek(pos)
        return result

    return wrapper


class BasicOffsetTable:
    r"""Helper class for manipulating the Basic Offset Table"""

    PREFIX: ClassVar[bytes] = b"\xFE\xFF\x00\xE0"
    DELIM: ClassVar[bytes] = b"\xFF\xFE\xE0\xDD"

    def __init__(self, fp: DicomBytesIO):
        if not isinstance(fp, DicomBytesIO):
            raise TypeError(type(fp))
        self.fp = copy(fp)
        self.fp.is_little_endian = True

        if not self.is_present(self.fp):
            raise ValueError("A basic offset table was not present")

        # trim the input stream to just the BOT
        size = self.total_length
        self.fp.seek(START)
        trimmed = self.fp.read(size)
        self.fp = DicomBytesIO(trimmed)
        self.fp.is_little_endian = True
        assert self.is_present(self.fp)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BasicOffsetTable):
            return False
        return self.buffer == other.buffer

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(total_length={self.total_length}, frames={self.num_frames})"

    @restore_pointer
    def __len__(self) -> int:
        r"""The value of the length component of the BOT"""
        self.fp.seek(LENGTH)
        return self.fp.read_UL()

    @property
    def buffer(self) -> bytes:
        r"""The raw bytes of this BOT"""
        # type errors when using restore_pointer, so handle it manually here
        pos = self.fp.tell()
        self.fp.seek(START)
        result = self.fp.read()
        self.fp.seek(pos)
        return result

    @property
    def total_length(self) -> int:
        r"""The total number of bytes in this BOT"""
        return len(self) + 2 * BYTES_PER_LONG

    @property
    def num_frames(self) -> int:
        r"""The total number of frames in this BOT"""
        return len(self) // 4

    @restore_pointer
    def __iter__(self) -> Iterator[int]:
        r"""Iterator over frame offsets"""
        steps = len(self) // 4
        self.fp.seek(OFFSETS)
        for i in range(steps):
            yield self.fp.read_UL()

    @property
    def is_valid(self) -> bool:
        r"""Checks if this basic offset table is valid.
        A basic offset table should have a length divisible by 4, and the frame
        offsets should be monotonically increasing.
        """
        return len(self) % BYTES_PER_LONG == 0 and list(self) == sorted(set(self))

    @classmethod
    @prepare_stream
    def is_present(cls, stream: DicomBytesIO) -> bool:
        r"""Checks if a stream contains a BOT"""
        pos = stream.tell()
        stream.seek(PREFIX)
        prefix = stream.read(BYTES_PER_LONG)
        stream.seek(pos)
        return prefix == cls.PREFIX

    @classmethod
    def from_offsets(cls, offsets: Sequence[int]) -> "BasicOffsetTable":
        r"""Create a BOT from a sequence of frame offsets"""
        stream = bytearray()
        # prefix and length
        stream.extend(cls.PREFIX)
        stream.extend(pack("<I", BYTES_PER_LONG * len(offsets)))
        for offset in offsets:
            stream.extend(pack("<I", offset))
        return cls.from_stream(bytes(stream))

    @classmethod
    @prepare_stream
    def from_stream(cls, stream: DicomBytesIO) -> "BasicOffsetTable":
        r"""Create a BOT from a byte stream"""
        stream.seek(START)
        return cls(stream)

    @classmethod
    def default(cls) -> "BasicOffsetTable":
        r"""Create a default (empty) BOT"""
        data = b"".join([cls.PREFIX, b"\x00\x00\x00\x00"])
        return cls.from_stream(data)

    @classmethod
    @prepare_stream
    def remove_from(cls, stream: DicomBytesIO) -> bytes:
        r"""Removes the BOT from a stream"""
        pos = stream.tell()
        if cls.is_present(stream):
            bot = cls.from_stream(stream)
            stream.seek(bot.total_length)
        else:
            stream.seek(START)
        result = stream.read()
        stream.seek(pos)
        return result

    @prepare_stream
    def prepend_to(self, stream: DicomBytesIO) -> bytes:
        r"""Prepends this BOT to the start of a stream"""
        stream.seek(START)
        buffer = cast(bytes, self.buffer)
        return b"".join([buffer, stream.read()])

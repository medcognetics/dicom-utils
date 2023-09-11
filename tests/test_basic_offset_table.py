#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pydicom.filebase import DicomBytesIO

from dicom_utils.basic_offset_table import BYTES_PER_LONG, LENGTH, BasicOffsetTable
from dicom_utils.dicom import Dicom


@pytest.fixture
def dummy_dicom_factory(mocker):
    def func(pixels):
        m = mocker.MagicMock(Dicom)
        m.PixelData = pixels
        return m

    return func


@pytest.fixture(params=["bytes", "dicombytesio", "dicom"])
def stream_type_factory(dummy_dicom_factory, request):
    stream_type = request.param

    def func(stream):
        if stream_type == "dicombytesio":
            stream = DicomBytesIO(stream)
        elif stream_type == "dicom":
            stream = dummy_dicom_factory(stream)
        elif stream_type != "bytes":
            raise ValueError(stream_type)
        return stream

    return func


class TestBasicOffsetTable:
    @pytest.mark.parametrize(
        "offsets",
        [
            [0, 1000],
            [0, 1000, 2000],
        ],
    )
    def test_from_offsets(self, offsets):
        bot = BasicOffsetTable.from_offsets(offsets)
        assert len(bot) == BYTES_PER_LONG * len(offsets)
        assert list(bot) == offsets

    def test_default(self):
        bot = BasicOffsetTable.default()
        assert bot.num_frames == 0
        assert len(bot) == 0
        assert bot.total_length == 2 * BYTES_PER_LONG
        assert not list(bot)

    def test_repr(self):
        bot = BasicOffsetTable.default()
        s = repr(bot)
        assert isinstance(s, str)
        assert f"total_length={2*BYTES_PER_LONG}" in s
        assert "frames=0" in s

    @pytest.mark.parametrize(
        "offsets",
        [
            [0, 1000],
            [0, 1000, 2000],
        ],
    )
    def test_from_stream(self, offsets, stream_type_factory):
        # use from_offsets to simplify setup
        src = BasicOffsetTable.from_offsets(offsets)
        needs_trimming = b"".join([b"\x00"] * 4)
        stream = b"".join([src.buffer, needs_trimming])
        stream = stream_type_factory(stream)
        bot = BasicOffsetTable.from_stream(stream)
        assert bot == src

    @pytest.mark.parametrize(
        "offsets",
        [
            [0, 1000],
            [0, 1000, 2000],
        ],
    )
    def test_total_length(self, offsets):
        bot = BasicOffsetTable.from_offsets(offsets)
        assert bot.total_length == BYTES_PER_LONG * (len(offsets) + 2)

    @pytest.mark.parametrize(
        "offsets",
        [
            [0, 1000],
            [0, 1000, 2000],
        ],
    )
    def test_num_frames(self, offsets):
        bot = BasicOffsetTable.from_offsets(offsets)
        assert bot.num_frames == len(offsets)

    @pytest.mark.parametrize(
        "offsets, exp",
        [
            pytest.param([0, 1000], True),
            pytest.param([0, 1000, 2000], True),
            pytest.param([0, 0, 1000], False),
            pytest.param([0, 1000, 1000], False),
            pytest.param([0, 1000, 100], False),
        ],
    )
    def test_is_valid(self, offsets, exp):
        bot = BasicOffsetTable.from_offsets(offsets)
        assert bot.is_valid == exp

    def test_is_valid_mod_length(self):
        bot = BasicOffsetTable.from_offsets([0, 1000])
        bot.fp.seek(LENGTH)
        bot.fp.write_UL(val=3)  # type: ignore
        assert not bot.is_valid

    @pytest.mark.parametrize(
        "offsets",
        [
            [0, 1000],
            [0, 1000, 2000],
        ],
    )
    def test_remove_from(self, offsets, stream_type_factory):
        src = BasicOffsetTable.from_offsets(offsets)
        extras = b"".join([b"\x00"] * 4)
        stream = b"".join([src.buffer, extras])
        stream = stream_type_factory(stream)
        result = BasicOffsetTable.remove_from(stream)
        assert result == extras

    @pytest.mark.parametrize(
        "offsets",
        [
            [0, 1000],
            [0, 1000, 2000],
        ],
    )
    def test_prepend_to(self, offsets, stream_type_factory):
        bot = BasicOffsetTable.from_offsets(offsets)
        extras = b"".join([b"\x00"] * 4)
        stream = stream_type_factory(extras)
        result = bot.prepend_to(stream)
        assert BasicOffsetTable.from_stream(result) == bot
        assert len(result) == bot.total_length + len(extras)

    @pytest.mark.parametrize(
        "offsets,val,exp",
        [
            pytest.param([0, 1000], 0, True),
            pytest.param([0, 1000, 2000], 2000, True),
            pytest.param([0, 1000], -1, False),
            pytest.param([0, 1000, 2000], 3000, False),
        ],
    )
    def test_contains(self, offsets, val, exp):
        bot = BasicOffsetTable.from_offsets(offsets)
        assert (val in bot) == exp

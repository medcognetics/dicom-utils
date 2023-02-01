#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shutil
from dataclasses import fields, replace
from io import IOBase
from os import PathLike
from pathlib import Path
from typing import Any, List, Optional

import pydicom
import pytest
from pydicom import DataElement, Sequence
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue
from pydicom.uid import AllTransferSyntaxes, SecondaryCaptureImageStorage

from dicom_utils.container import FileRecord
from dicom_utils.container.record import (
    STANDARD_MAMMO_VIEWS,
    DicomFileRecord,
    DicomImageFileRecord,
    MammogramFileRecord,
    ModalityHelper,
    StandardizedFilename,
)
from dicom_utils.dicom_factory import DicomFactory
from dicom_utils.tags import Tag
from dicom_utils.types import Laterality, MammogramType, PixelSpacing, ViewPosition, get_value


class TestStandardizedFilename:
    @pytest.mark.parametrize(
        "inp, exp",
        [
            pytest.param(Path("foo_1.txt"), StandardizedFilename("foo_1.txt")),
            pytest.param(Path("bar_1234"), StandardizedFilename("bar_1234")),
            pytest.param(Path("foo.txt"), StandardizedFilename("foo_1.txt")),
        ],
    )
    def test_create(self, inp, exp):
        p = StandardizedFilename(inp)
        assert isinstance(p, StandardizedFilename)
        assert isinstance(p, Path)
        assert p == exp

    @pytest.mark.parametrize(
        "inp,exp",
        [
            pytest.param(StandardizedFilename("foo_1.txt"), "1"),
            pytest.param(StandardizedFilename("bar_2.txt"), "2"),
            pytest.param(StandardizedFilename("baz_3"), "3"),
            pytest.param(StandardizedFilename("bar_1234.txt"), "1234"),
            pytest.param(StandardizedFilename("bar_1234"), "1234"),
        ],
    )
    def test_file_id(self, inp, exp):
        assert inp.file_id == exp

    @pytest.mark.parametrize(
        "inp,id,exp",
        [
            pytest.param(StandardizedFilename("foo.txt"), "1", StandardizedFilename("foo_1.txt")),
            pytest.param(StandardizedFilename("foo_2.txt"), "1", StandardizedFilename("foo_1.txt")),
            pytest.param(StandardizedFilename("bar"), "1", StandardizedFilename("bar_1")),
            pytest.param(StandardizedFilename("bar_123.txt"), "234", StandardizedFilename("bar_234.txt")),
        ],
    )
    def test_with_file_id(self, inp, id, exp):
        assert inp.with_file_id(id) == exp

    @pytest.mark.parametrize(
        "inp,exp",
        [
            pytest.param(StandardizedFilename("ffdm_1.dcm"), "ffdm"),
            pytest.param(StandardizedFilename("ffdm_mag_spot_1.dcm"), "ffdm_mag_spot"),
            pytest.param(StandardizedFilename("ffdm_mag_spot_1"), "ffdm_mag_spot"),
        ],
    )
    def test_prefix(self, inp, exp):
        assert inp.prefix == exp

    @pytest.mark.parametrize(
        "inp,val,exp",
        [
            pytest.param(StandardizedFilename("file_1.dcm"), ["ffdm"], StandardizedFilename("ffdm_1.dcm")),
            pytest.param(StandardizedFilename("file_2.dcm"), ["ffdm", "spot"], StandardizedFilename("ffdm_spot_2.dcm")),
        ],
    )
    def test_with_prefix(self, inp, val, exp):
        assert inp.with_prefix(*val) == exp

    @pytest.mark.parametrize(
        "inp,mod,exp",
        [
            pytest.param(StandardizedFilename("ffdm_1.dcm"), "spot", StandardizedFilename("ffdm_spot_1.dcm")),
            pytest.param(StandardizedFilename("ffdm_mag_2.dcm"), "spot", StandardizedFilename("ffdm_mag_spot_2.dcm")),
        ],
    )
    def test_add_modifier(self, inp, mod, exp):
        assert inp.add_modifier(mod) == exp


class TestFileRecord:
    @pytest.fixture
    def record_factory(self, tmp_path):
        def func(filename: PathLike = Path("foo.txt")):
            filename = Path(tmp_path, filename)
            filename.parent.mkdir(exist_ok=True, parents=True)
            filename.touch()
            record = FileRecord(filename)
            return record

        return func

    def test_repr(self, record_factory):
        rec = record_factory("a.txt")
        s = repr(rec)
        assert isinstance(s, str)
        assert rec.__class__.__name__ in s

    def test_hash(self, record_factory):
        rec1 = record_factory("foo.txt")
        rec2 = record_factory("foo.txt")
        rec3 = record_factory("bar.txt")
        assert hash(rec1) == hash(rec2)
        assert hash(rec1) != hash(rec3)

    def test_eq(self, record_factory):
        rec1 = record_factory("foo.txt")
        rec2 = record_factory("foo.txt")
        rec3 = record_factory("bar.txt")
        assert rec1 == rec2
        assert rec1 != rec3

    def test_compare(self, record_factory):
        rec1 = record_factory("a.txt")
        rec2 = record_factory("b.txt")
        rec3 = record_factory("c.txt")
        assert sorted([rec3, rec2, rec1]) == [rec1, rec2, rec3]

    def test_has_uid(self, record_factory):
        rec = record_factory("a.txt")
        assert rec.has_uid

    def test_get_uid(self, record_factory):
        rec = record_factory("a.txt")
        assert rec.get_uid() == str(rec.path.stem)

    @pytest.mark.parametrize(
        "path,target,exp",
        [
            pytest.param("foo/bar.txt", "foo/", "bar.txt"),
            pytest.param("foo/bar/baz.txt", "foo/baz/bar.txt", "../../bar/baz.txt"),
            pytest.param("foo/bar/baz.txt", "foo/baz/", "../bar/baz.txt"),
        ],
    )
    def test_relative_to(self, tmp_path, path, target, exp, record_factory):
        target = Path(tmp_path, target)
        rec = record_factory(path)
        relative_rec = rec.relative_to(target)
        assert type(relative_rec) == type(rec)
        assert relative_rec.path == Path(exp)

    @pytest.mark.parametrize(
        "path,target,exp",
        [
            pytest.param("foo/bar.txt", "foo/baz.txt", True),
            pytest.param("foo/baz/bar.txt", "foo/baz.txt", False),
            pytest.param("foo/baz/bar.txt", "foo/bar/baz.txt", False),
            pytest.param("foo.txt", "foo.txt", True),
        ],
    )
    def shares_directory_with(self, path1, path2, exp, record_factory):
        rec1 = record_factory(path1)
        rec2 = record_factory(path2)
        assert rec1.shares_directory_with(rec2) == exp
        assert rec2.shares_directory_with(rec1) == exp

    def test_present_fields(self, record_factory):
        rec = record_factory()
        actual = dict(rec.present_fields())
        expected = {field.name: value for field in fields(rec) if (value := getattr(rec, field.name)) != field.default}
        assert actual == expected

    def test_file_size(self, record_factory):
        rec = record_factory()
        actual = rec.file_size
        expected = rec.path.stat().st_size
        assert actual == expected

    def test_is_compressed(self, record_factory):
        rec = record_factory()
        assert not rec.is_compressed

    @pytest.mark.parametrize(
        "path,file_id,exp",
        [
            pytest.param("foo/bar.txt", None, "bar_bar.txt"),
            pytest.param("foo/bar.txt", "2", "bar_2.txt"),
            pytest.param("foo/bar.txt", 1, "bar_1.txt"),
        ],
    )
    def test_standardized_filename(self, path, file_id, exp, record_factory):
        rec = record_factory(path)
        actual = rec.standardized_filename(file_id)
        assert isinstance(actual, StandardizedFilename)
        assert actual == Path(exp)

    def test_read(self, record_factory):
        rec = record_factory()
        stream = rec.__class__.read(rec.path)
        assert isinstance(stream, IOBase)

    @pytest.mark.parametrize(
        "path,symlink",
        [
            pytest.param("foo/bar/baz.txt", "foo/link/baz.txt"),
            pytest.param("1/2/baz.txt", "2/1/baz.txt"),
            pytest.param("1/2/baz.txt", "2/foo.txt"),
        ],
    )
    def test_to_symlink(self, tmp_path, path, symlink, record_factory):
        rec = record_factory(path)
        symlink = Path(tmp_path, symlink)
        symlink_rec = rec.to_symlink(symlink)
        assert symlink_rec.path.is_symlink()
        assert symlink_rec.path.resolve() == rec.path

    def test_to_dict(self, record_factory):
        rec = record_factory("a.txt")
        rec_dict = rec.to_dict()
        assert rec_dict["record_type"] == rec.__class__.__name__
        assert rec_dict["path"] == str(rec.path.absolute())
        assert rec_dict["resolved_path"] == str(rec.path.resolve().absolute())


def make_view_modifier_code(meaning: str) -> Dataset:
    vc = Dataset()
    vc[Tag.CodeMeaning] = DataElement(Tag.CodeMeaning, "ST", meaning)
    return vc


class TestDicomFileRecord(TestFileRecord):
    @pytest.fixture
    def record_factory(self, tmp_path, dicom_file):
        def func(filename: PathLike = Path("foo.txt"), **kwargs):
            filename = Path(tmp_path, filename)
            filename.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(dicom_file, filename)
            record = DicomFileRecord.from_file(filename, **kwargs)
            return record

        return func

    def test_from_file_assigns_tags(self, record_factory):
        rec = record_factory()
        dcm = pydicom.dcmread(rec.path, stop_before_pixels=True)
        for field in fields(rec):
            tag = getattr(Tag, field.name, None)
            # shortcut for non-tag fields
            if tag is None:
                continue
            value = getattr(rec, field.name, None)
            expected = get_value(dcm, tag, None, try_file_meta=True)
            # shortcut for modality so subclassing tests that override modality
            # won't fail
            if tag == Tag.Modality:
                assert value is not None
            elif expected is not None:
                assert value == expected

    def test_from_file_only_reads_once(self, mocker, record_factory):
        spy = mocker.spy(pydicom, "dcmread")
        proto = record_factory()
        spy.reset_mock()
        proto.__class__.from_file(proto.path)
        spy.assert_called_once()

    @pytest.mark.parametrize("file_exists", [True, False])
    def test_from_dicom(self, mocker, record_factory, file_exists):
        proto = record_factory()
        with pydicom.dcmread(proto.path) as dcm:
            if not file_exists:
                proto.path.unlink()
                assert not proto.path.is_file()
            result = proto.__class__.from_dicom(proto.path, dcm)
        assert result == proto

    @pytest.mark.parametrize(
        "sop,series,exp",
        [
            pytest.param("1.2.345", "2.3.456", True),
            pytest.param("1.2.345", None, True),
            pytest.param(None, "2.3.456", True),
            pytest.param(None, None, False),
        ],
    )
    def test_has_uid(self, sop, series, exp, record_factory):
        rec = record_factory()
        rec = rec.replace(
            SOPInstanceUID=sop,
            SeriesInstanceUID=series,
        )
        assert rec.has_uid == exp

    @pytest.mark.parametrize(
        "sop,series,prefer_sop,exp",
        [
            pytest.param("1.2.345", "2.3.456", False, "2.3.456"),
            pytest.param("1.2.345", "2.3.456", True, "1.2.345"),
            pytest.param("1.2.345", None, False, "1.2.345"),
            pytest.param(None, "2.3.456", True, "2.3.456"),
            pytest.param(None, None, None, None, marks=pytest.mark.xfail(raises=AttributeError)),
        ],
    )
    def test_get_uid(self, sop, series, prefer_sop, exp, record_factory):
        rec = record_factory()
        rec = rec.replace(
            SOPInstanceUID=sop,
            SeriesInstanceUID=series,
        )
        uid = rec.get_uid(prefer_sop)
        assert uid == exp

    @pytest.mark.parametrize(
        "path,modality,file_id,exp",
        [
            pytest.param("foo/bar.dcm", "CT", None, "ct_1-2-345.dcm"),
            pytest.param("foo/bar.dcm", "US", "2", "us_2.dcm"),
            pytest.param("foo/bar.dcm", "MG", 1, "mg_1.dcm"),
        ],
    )
    def test_standardized_filename(self, path, modality, file_id, exp, record_factory):
        rec = record_factory(path)
        sop = "1.2.345"
        rec = rec.replace(SOPInstanceUID=sop, Modality=modality)
        actual = rec.standardized_filename(file_id)
        assert isinstance(actual, StandardizedFilename)
        assert actual == Path(exp)

    def test_read(self, mocker, record_factory):
        spy = mocker.spy(pydicom, "dcmread")
        rec = record_factory()
        spy.reset_mock()
        stream = rec.__class__.read(rec.path)
        assert isinstance(stream, pydicom.FileDataset)
        spy.assert_called_once()
        assert spy.mock_calls[0].kwargs["stop_before_pixels"]

    @pytest.mark.parametrize(
        "study_date,exp",
        [
            pytest.param("20200101", 2020),
            pytest.param("00010101", 1),
            pytest.param("", None),
            pytest.param("fooo", None),
            pytest.param("10", None),
        ],
    )
    def test_year(self, study_date, exp, record_factory):
        rec = record_factory()
        rec = rec.replace(StudyDate=study_date)
        assert rec.year == exp

    def test_to_dict(self, record_factory):
        rec = record_factory("a.dcm")
        rec_dict = rec.to_dict()
        assert rec_dict["record_type"] == rec.__class__.__name__
        assert rec_dict["path"] == str(rec.path.absolute())
        assert rec_dict["resolved_path"] == str(rec.path.resolve().absolute())
        assert rec_dict["Modality"] == rec.Modality

    def test_hash(self, record_factory):
        rec1 = record_factory("a.dcm")
        rec2 = record_factory("b.dcm")
        rec3 = record_factory("a.dcm")
        rec1 = rec1.replace(SOPInstanceUID="123")
        rec2 = rec2.replace(SOPInstanceUID="123")
        assert hash(rec1) == hash(rec2)
        assert hash(rec1) != hash(rec3)

    @pytest.mark.parametrize("has_uid", [False, True])
    def test_eq(self, record_factory, has_uid):
        rec1 = record_factory("a.dcm")
        rec2 = record_factory("b.dcm")
        rec3 = record_factory("a.dcm")
        rec1 = rec1.replace(SOPInstanceUID="123" if has_uid else None)
        rec2 = rec2.replace(SOPInstanceUID="123" if has_uid else None)
        rec3 = rec3.replace(SOPInstanceUID="234" if has_uid else None)

        if has_uid:
            assert rec1 == rec2
            assert rec1 != rec3
        else:
            assert rec1 != rec2
            assert rec1 == rec3

    @pytest.mark.parametrize(
        "addr,name,ts,exp",
        [
            (None, None, None, None),
            ("foo", None, None, "foo"),
            (None, "foo", None, "foo"),
            (None, None, "foo", "foo"),
            ("foo", "bar", "baz", "foo"),
            (None, "bar", "baz", "baz"),
        ],
    )
    def test_site(self, addr, name, ts, exp, record_factory):
        rec = record_factory("foo.dcm")
        rec = rec.replace(
            InstitutionAddress=addr,
            InstitutionName=name,
            TreatmentSite=ts,
        )
        assert rec.site == exp


class TestDicomImageFileRecord(TestDicomFileRecord):
    @pytest.fixture
    def record_factory(self, tmp_path, dicom_file):
        def func(
            filename: PathLike = Path("foo.txt"),
            Rows: Optional[int] = 128,
            Columns: Optional[int] = 128,
            NumberOfFrames: Optional[int] = None,
            Modality: Optional[str] = "CT",
            view_modifier_code: Optional[str] = None,
            PixelSpacing: Optional[Any] = MultiValue(str, [0.661468, 0.661468]),
            ImagerPixelSpacing: Optional[Any] = None,
            **kwargs,
        ):
            filename = Path(tmp_path, filename)
            filename.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(dicom_file, filename)
            record = DicomImageFileRecord.from_file(filename, **kwargs)

            if view_modifier_code is not None:
                vc = make_view_modifier_code(view_modifier_code)
                view_modifier_code_seq = Sequence([vc])
            else:
                view_modifier_code_seq = None

            record = record.replace(
                Rows=Rows,
                Columns=Columns,
                NumberOfFrames=NumberOfFrames,
                Modality=kwargs.get("modality", Modality),
                ViewModifierCodeSequence=view_modifier_code_seq,
                PixelSpacing=PixelSpacing,
                ImagerPixelSpacing=ImagerPixelSpacing,
            )
            return record

        return func

    @pytest.mark.parametrize(
        "rows,columns,nf,exp",
        [
            pytest.param(100, 100, 100, True),
            pytest.param(100, 100, None, True),
            pytest.param(None, 100, None, False),
            pytest.param(100, None, None, False),
            pytest.param(None, None, None, False),
        ],
    )
    def test_is_valid_image(self, rows, columns, nf, exp, record_factory):
        rec = record_factory(Rows=rows, Columns=columns, NumberOfFrames=nf)
        assert rec.is_valid_image == exp

    @pytest.mark.parametrize(
        "tsuid,exp",
        [pytest.param(tsuid, tsuid.is_compressed) for tsuid in AllTransferSyntaxes],
    )
    def test_is_compressed(self, tsuid, exp, record_factory):
        rec = record_factory()
        rec = rec.replace(TransferSyntaxUID=tsuid)
        assert rec.is_compressed == exp

    @pytest.mark.parametrize(
        "rows,columns,nf,exp",
        [
            pytest.param(100, 100, 100, True),
            pytest.param(100, 100, 1, False),
            pytest.param(100, 100, None, False),
            pytest.param(None, 100, 100, False),
            pytest.param(None, 100, None, False),
            pytest.param(100, None, None, False),
            pytest.param(None, None, None, False),
        ],
    )
    def test_is_volume(self, rows, columns, nf, exp, record_factory):
        rec = record_factory(Rows=rows, Columns=columns, NumberOfFrames=nf)
        assert rec.is_volume == exp

    @pytest.mark.parametrize(
        "code,exp",
        [
            pytest.param("spot compression", False),
            pytest.param("magnified", True),
            pytest.param("magnification", True),
        ],
    )
    def test_is_magnified(self, code, exp, record_factory):
        rec = record_factory(view_modifier_code=code)
        assert rec.is_magnified == exp

    @pytest.mark.parametrize(
        "spacing,imager_spacing,exp",
        [
            pytest.param("[0.01, 0.01]", None, PixelSpacing(0.01, 0.01)),
            pytest.param(None, "[0.01, 0.01]", PixelSpacing(0.01, 0.01)),
            pytest.param(MultiValue(str, [0.02, 0.02]), None, PixelSpacing(0.02, 0.02)),
            pytest.param(None, MultiValue(str, [0.02, 0.02]), PixelSpacing(0.02, 0.02)),
            pytest.param(None, None, None),
        ],
    )
    def test_pixel_spacing(self, spacing, imager_spacing, exp, record_factory):
        rec = record_factory(PixelSpacing=spacing, ImagerPixelSpacing=imager_spacing)
        assert rec.pixel_spacing == exp


class TestMammogramFileRecord(TestDicomFileRecord):
    @pytest.fixture
    def record_factory(self, tmp_path, dicom_file):
        def func(
            filename: PathLike = Path("foo.dcm"),
            Rows: Optional[int] = 128,
            Columns: Optional[int] = 128,
            NumberOfFrames: Optional[int] = None,
            Modality: Optional[str] = "MG",
            view_modifier_code: Optional[str] = "medio-lateral oblique",
            PixelSpacing: Optional[Any] = MultiValue(str, [0.661468, 0.661468]),
            ImagerPixelSpacing: Optional[Any] = None,
            laterality: Optional[Laterality] = Laterality.LEFT,
            view_position: Optional[ViewPosition] = ViewPosition.MLO,
            mammogram_type: Optional[MammogramType] = MammogramType.FFDM,
            **kwargs,
        ):
            filename = Path(tmp_path, filename)
            filename.parent.mkdir(exist_ok=True, parents=True)
            with pydicom.dcmread(dicom_file) as dcm:
                dcm.Modality = "MG"
                dcm.save_as(filename)
            record = MammogramFileRecord.from_file(filename, **kwargs)

            if view_modifier_code is not None:
                vc = make_view_modifier_code(view_modifier_code)
                view_modifier_code_seq = Sequence([vc])
            else:
                view_modifier_code_seq = None

            if laterality is not None:
                record = record.replace(laterality=laterality)
            if view_position is not None:
                record = record.replace(view_position=view_position)
            if mammogram_type is not None:
                record = record.replace(mammogram_type=mammogram_type)

            record = record.replace(
                Rows=Rows,
                Columns=Columns,
                NumberOfFrames=NumberOfFrames,
                Modality=kwargs.get("modality", Modality),
                ViewModifierCodeSequence=view_modifier_code_seq,
                PixelSpacing=PixelSpacing,
                ImagerPixelSpacing=ImagerPixelSpacing,
            )
            return record

        return func

    @pytest.mark.parametrize(
        "dtype,attr",
        [
            pytest.param(Laterality, "laterality"),
            pytest.param(ViewPosition, "view_position"),
            pytest.param(MammogramType, "mammogram_type"),
        ],
    )
    def test_from_file_assigns_mammogram_attrs(self, mocker, dtype, attr, record_factory):
        m = mocker.patch.object(dtype, "from_dicom", spec_set=dtype)
        rec = record_factory(**{attr: None})
        m.assert_called_once()
        assert getattr(rec, attr) == m()

    @pytest.mark.parametrize(
        "paddle,code,view_pos,exp",
        [
            pytest.param("SPOT", None, None, True),
            pytest.param("SPOT COMPRESSION", None, None, True),
            pytest.param(None, None, None, False),
            pytest.param(None, "spot compression", None, True),
            pytest.param(None, None, "CCSpot", True),
        ],
    )
    def test_is_spot_compression(self, paddle, code, view_pos, exp, record_factory):
        record = record_factory()
        seq = Sequence([make_view_modifier_code(code)]) if code is not None else None
        record = replace(
            record,
            Modality="MG",
            PaddleDescription=paddle,
            ViewModifierCodeSequence=seq,
            ViewPosition=view_pos,
        )
        assert record.is_spot_compression == exp

    @pytest.mark.parametrize(
        "code,exp",
        [
            pytest.param("implant displaced", True),
            pytest.param("magnified", False),
            pytest.param("", False),
        ],
    )
    def test_is_implant_displaced(self, code, exp, record_factory):
        record = record_factory()
        seq = Sequence([make_view_modifier_code(code)]) if code is not None else None
        record = replace(
            record,
            Modality="MG",
            ViewModifierCodeSequence=seq,
        )
        assert record.is_implant_displaced == exp

    @pytest.mark.parametrize(
        "view_pos,exp",
        [
            pytest.param(ViewPosition.MLO, True),
            pytest.param(ViewPosition.CC, True),
            *[pytest.param(x, False) for x in ViewPosition if x not in (ViewPosition.MLO, ViewPosition.CC)],
        ],
    )
    def test_is_standard_mammo_view(self, view_pos, exp, record_factory):
        record = record_factory(view_position=view_pos)
        assert record.is_standard_mammo_view == exp

    @pytest.mark.parametrize(
        "spot,mag,secondary,for_proc,cad,stereo,exp",
        [
            pytest.param(False, False, False, False, False, False, True),
            pytest.param(True, False, False, False, False, False, False),
            pytest.param(False, True, False, False, False, False, False),
            pytest.param(False, False, True, False, False, False, False),
            pytest.param(False, False, False, True, False, False, False),
            pytest.param(False, False, False, False, True, False, False),
            pytest.param(False, False, False, False, False, True, False),
        ],
    )
    def test_is_standard_mammo_view_modifiers(
        self, mocker, spot, mag, secondary, for_proc, cad, stereo, exp, record_factory
    ):
        record = record_factory(view_position=ViewPosition.MLO)
        if spot:
            object.__setattr__(record, "is_spot_compression", True)
        if mag:
            object.__setattr__(record, "is_magnified", True)
        if secondary:
            # We can't set `is_secondary_capture` directly since it's not a cached_property
            object.__setattr__(record, "SOPClassUID", SecondaryCaptureImageStorage)
        if for_proc:
            object.__setattr__(record, "is_for_processing", True)
        if cad:
            object.__setattr__(record, "is_cad", True)
        if stereo:
            object.__setattr__(record, "is_stereo", True)
        assert record.is_standard_mammo_view == exp

    @pytest.mark.parametrize("secondary_capture", [False, True])
    def test_is_complete_mammo_case(self, secondary_capture, record_factory):
        record = record_factory()
        if secondary_capture:
            record = record.replace(SOPClassUID=SecondaryCaptureImageStorage)
        # should be incomplete until after this loop
        records: List[MammogramFileRecord] = []
        for laterality, view_pos in STANDARD_MAMMO_VIEWS:
            assert not MammogramFileRecord.is_complete_mammo_case(records)
            rec = replace(record, laterality=laterality, view_position=view_pos)
            records.append(rec)

        actual = MammogramFileRecord.is_complete_mammo_case(records)
        expected = not secondary_capture
        assert actual == expected

    @pytest.mark.parametrize(
        "mtype,spot,mag,id,laterality,view_pos,uid,secondary,for_proc,stereo,exp",
        [
            pytest.param(
                MammogramType.FFDM,
                False,
                False,
                False,
                Laterality.LEFT,
                ViewPosition.MLO,
                "1",
                False,
                False,
                False,
                "ffdm_lmlo_1.dcm",
            ),
            pytest.param(
                MammogramType.FFDM,
                False,
                False,
                False,
                Laterality.RIGHT,
                ViewPosition.CC,
                "2",
                False,
                False,
                False,
                "ffdm_rcc_2.dcm",
            ),
            pytest.param(
                MammogramType.SYNTH,
                True,
                True,
                True,
                Laterality.RIGHT,
                ViewPosition.XCCL,
                "1",
                False,
                False,
                False,
                "synth_rxccl_spot_mag_id_1.dcm",
            ),
            pytest.param(
                MammogramType.FFDM,
                False,
                False,
                False,
                Laterality.UNKNOWN,
                ViewPosition.UNKNOWN,
                "2",
                False,
                False,
                False,
                "ffdm_2.dcm",
            ),
            pytest.param(
                MammogramType.SYNTH,
                True,
                True,
                True,
                Laterality.RIGHT,
                ViewPosition.XCCL,
                "1",
                True,
                True,
                False,
                "synth_rxccl_secondary_proc_spot_mag_id_1.dcm",
            ),
            pytest.param(
                MammogramType.FFDM,
                False,
                False,
                False,
                Laterality.RIGHT,
                ViewPosition.CC,
                "1",
                False,
                False,
                True,
                "ffdm_rcc_stereo_1.dcm",
            ),
        ],
    )
    def test_standardized_filename(
        self, mtype, spot, mag, id, laterality, view_pos, uid, secondary, for_proc, stereo, exp, record_factory
    ):
        seq = []
        if spot:
            seq.append(make_view_modifier_code("spot compression"))
        if mag:
            seq.append(make_view_modifier_code("magnification"))
        if id:
            seq.append(make_view_modifier_code("implant displaced"))
        seq = Sequence(seq)
        record = record_factory(
            mammogram_type=mtype,
            laterality=laterality,
            view_position=view_pos,
        )
        record = record.replace(
            mammogram_type=mtype,
            ViewModifierCodeSequence=seq,
            laterality=laterality,
            view_position=view_pos,
            SOPClassUID=SecondaryCaptureImageStorage if secondary else None,
            PresentationIntentType="FOR PROCESSING" if for_proc else None,
            PerformedProcedureStepDescription="Stereo, CC" if stereo else None,
        )
        actual = record.standardized_filename(uid)
        assert isinstance(actual, StandardizedFilename)
        assert actual == Path(exp)

    @pytest.mark.parametrize(
        "body_part,study,series,force,exp",
        [
            pytest.param("BREAST", "", "", False, "MG"),
            pytest.param("", "MAMMO SCREEN", "", False, "MG"),
            pytest.param("", "", "MAMMO SCREEN", False, "MG"),
            pytest.param("", "", "", True, "MG"),
        ],
    )
    def test_modality_helper(self, tmp_path, body_part, series, study, force, exp):
        dcm = DicomFactory()(
            Modality="CT",
            SeriesDescription=series,
            StudyDescription=study,
            BodyPartExamined=body_part,
        )
        path = Path(tmp_path, "test.dcm")
        dcm.save_as(path)
        helpers = [ModalityHelper(force=force)]
        rec = MammogramFileRecord.from_file(path, helpers=helpers)
        assert rec.Modality == exp

    @pytest.mark.parametrize(
        "l1,l2,exp",
        [
            pytest.param(None, None, Laterality.NONE),
            pytest.param(Laterality.LEFT, None, Laterality.LEFT),
            pytest.param(None, Laterality.RIGHT, Laterality.RIGHT),
            pytest.param(Laterality.LEFT, Laterality.RIGHT, Laterality.BILATERAL),
        ],
    )
    def test_collection_laterality(self, l1, l2, exp, record_factory):
        rec1 = record_factory(laterality=l1)
        rec2 = record_factory(laterality=l2)
        assert MammogramFileRecord.collection_laterality([rec1, rec2]) == exp

    @pytest.mark.parametrize(
        "override",
        ["MG", "US", "CR"],
    )
    def test_from_file_modality_override(self, override, record_factory):
        rec = record_factory(Modality=override)
        path = rec.path
        rec_type = type(rec)
        result = rec_type.from_file(path, overrides={"Modality": "MG"})
        assert result.Modality == "MG"

    @pytest.mark.parametrize(
        "modality",
        ["MG", "US", "CR"],
    )
    def test_from_dicom_modality_override(self, modality, record_factory):
        rec = record_factory()
        path = rec.path
        rec_type = type(rec)

        with pydicom.dcmread(path) as dcm:
            dcm.Modality = modality
            result1 = rec_type.from_dicom(path, pydicom.dcmread(path), overrides={"Modality": "MG"})
            result2 = rec_type.from_dicom(path, pydicom.dcmread(path), Modality="MG")
        assert result1.Modality == "MG"
        assert result2.Modality == "MG"

    @pytest.mark.parametrize(
        "tag,val,exp",
        [
            pytest.param(Tag.StudyDescription, "Stereo, LCC", True),
            pytest.param(Tag.SeriesDescription, "L CC Stereo Projection", True),
            pytest.param(Tag.PerformedProcedureStepDescription, "Stereo, LCC", True),
            pytest.param(None, "", False),
        ],
    )
    def test_is_stereo(self, tag, val, exp, record_factory):
        record = record_factory()
        if tag is not None:
            record = record.replace(**{tag.name: val})
        assert record.is_stereo == exp

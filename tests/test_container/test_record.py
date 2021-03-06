#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shutil
from dataclasses import fields, replace
from io import IOBase
from os import PathLike
from pathlib import Path
from typing import List, Optional

import pydicom
import pytest
from pydicom import DataElement, Sequence
from pydicom.dataset import Dataset
from pydicom.uid import AllTransferSyntaxes, SecondaryCaptureImageStorage

from dicom_utils.container import FileRecord
from dicom_utils.container.record import (
    STANDARD_MAMMO_VIEWS,
    DicomFileRecord,
    DicomImageFileRecord,
    MammogramFileRecord,
)
from dicom_utils.tags import Tag
from dicom_utils.types import DicomValueError, Laterality, MammogramType, ViewPosition, get_value


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
        spy.reset_mock()
        record_factory()
        spy.assert_called_once()

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
            pytest.param("foo/bar.dcm", "CT", None, "ct_1.2.345.dcm"),
            pytest.param("foo/bar.dcm", "US", "2", "us_2.dcm"),
            pytest.param("foo/bar.dcm", "MG", 1, "mg_1.dcm"),
        ],
    )
    def test_standardized_filename(self, path, modality, file_id, exp, record_factory):
        rec = record_factory(path)
        sop = "1.2.345"
        rec = rec.replace(SOPInstanceUID=sop, Modality=modality)
        actual = rec.standardized_filename(file_id)
        assert actual == Path(exp)

    def test_read(self, mocker, record_factory):
        spy = mocker.spy(pydicom, "dcmread")
        rec = record_factory()
        spy.reset_mock()
        stream = rec.__class__.read(rec.path)
        assert isinstance(stream, pydicom.FileDataset)
        spy.assert_called_once()
        assert spy.mock_calls[0].kwargs["stop_before_pixels"]

    @pytest.mark.parametrize("modality", ["CT", "MG", "US"])
    def test_modality_override(self, modality, record_factory):
        rec = record_factory(modality=modality)
        assert rec.Modality == modality

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
            laterality: Optional[Laterality] = Laterality.LEFT,
            view_position: Optional[ViewPosition] = ViewPosition.MLO,
            mammogram_type: Optional[MammogramType] = MammogramType.FFDM,
            **kwargs,
        ):
            filename = Path(tmp_path, filename)
            filename.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(dicom_file, filename)
            kwargs.setdefault("modality", Modality)
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
        "modality",
        [
            "MG",
            pytest.param("CT", marks=pytest.mark.xfail(raises=DicomValueError)),
            pytest.param("US", marks=pytest.mark.xfail(raises=DicomValueError)),
        ],
    )
    def test_modality_override(self, modality, record_factory):
        rec = record_factory(modality=modality)
        assert rec.Modality == modality

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

    @pytest.mark.parametrize("secondary_capture", [False, True])
    def test_is_complete_mammo_case(self, secondary_capture, record_factory):
        record = record_factory()
        if secondary_capture:
            record = record.replace(SOPClassUID=SecondaryCaptureImageStorage)
        # should be incomplete until after this loop
        records: List[MammogramFileRecord] = []
        for (laterality, view_pos) in STANDARD_MAMMO_VIEWS:
            assert not MammogramFileRecord.is_complete_mammo_case(records)
            rec = replace(record, laterality=laterality, view_position=view_pos)
            records.append(rec)

        actual = MammogramFileRecord.is_complete_mammo_case(records)
        expected = not secondary_capture
        assert actual == expected

    @pytest.mark.parametrize(
        "mtype,spot,mag,id,laterality,view_pos,uid,exp",
        [
            pytest.param(
                MammogramType.FFDM,
                False,
                False,
                False,
                Laterality.LEFT,
                ViewPosition.MLO,
                "1",
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
                "ffdm_2.dcm",
            ),
        ],
    )
    def test_standardized_filename(self, mtype, spot, mag, id, laterality, view_pos, uid, exp, record_factory):
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
        )
        assert record.standardized_filename(uid) == Path(exp)

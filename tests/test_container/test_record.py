#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import List

import pydicom
import pytest
from pydicom import DataElement, Sequence
from pydicom.dataset import Dataset
from pydicom.uid import SecondaryCaptureImageStorage

from dicom_utils.container import FileRecord, RecordCollection, record_iterator
from dicom_utils.container.record import STANDARD_MAMMO_VIEWS
from dicom_utils.tags import Tag
from dicom_utils.types import Laterality, MammogramType, ViewPosition


def make_view_modifier_code(meaning: str) -> Dataset:
    vc = Dataset()
    vc[Tag.CodeMeaning] = DataElement(Tag.CodeMeaning, "ST", meaning)
    return vc


class TestFileRecord:
    def test_create(self, dicom_file):
        rec = FileRecord.create(Path(dicom_file))
        dcm = pydicom.dcmread(dicom_file)
        assert rec.path == Path(dicom_file)
        assert rec.StudyInstanceUID == dcm.StudyInstanceUID
        assert rec.SeriesInstanceUID == dcm.SeriesInstanceUID
        assert rec.SOPInstanceUID == dcm.SOPInstanceUID
        assert rec.SOPClassUID == dcm.SOPClassUID
        assert rec.TransferSyntaxUID == dcm.file_meta.TransferSyntaxUID
        assert rec.Rows == dcm.Rows
        assert rec.Columns == dcm.Columns
        assert rec.NumberOfFrames == dcm.get("NumberOfFrames", None)
        assert rec.PhotometricInterpretation == dcm.PhotometricInterpretation
        assert rec.mammogram_type is None
        assert rec.ManufacturerModelName == dcm.ManufacturerModelName
        assert rec.SeriesDescription == dcm.get("SeriesDescription", None)
        assert rec.PatientName == dcm.get("PatientName", None)
        assert rec.PatientID == dcm.get("PatientID", None)
        assert rec.laterality == Laterality.UNKNOWN
        assert rec.view_position == ViewPosition.UNKNOWN

    @pytest.fixture
    def record(self, dicom_file):
        record = FileRecord.create(Path(dicom_file))
        vc = make_view_modifier_code("medio-lateral oblique")
        record = replace(
            record,
            Rows=100,
            Columns=100,
            Modality="MG",
            ViewModifierCodeSequence=Sequence([vc]),
            laterality=Laterality.LEFT,
            view_position=ViewPosition.MLO,
            mammogram_type=MammogramType.FFDM,
        )
        return record

    def test_hash(self, record):
        assert hash(record) == hash(record.path)

    def test_eq(self, record):
        rec2 = replace(record, path=Path("foo/bar.dcm"))
        rec3 = deepcopy(record)
        assert record != "dog"
        assert record != rec2
        assert record == rec3
        assert rec3 == record

    def test_file_size(self, record):
        assert record.file_size == record.path.stat().st_size

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
    def test_is_image(self, record, rows, columns, nf, exp):
        record = replace(record, Rows=rows, Columns=columns, NumberOfFrames=nf)
        assert record.is_image == exp

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
    def test_is_volume(self, record, rows, columns, nf, exp):
        record = replace(record, Rows=rows, Columns=columns, NumberOfFrames=nf)
        assert record.is_volume == exp

    @pytest.mark.parametrize(
        "rows,columns,modality,exp",
        [
            pytest.param(100, 100, "MG", True),
            pytest.param(None, 100, "MG", False),
            pytest.param(100, None, "MG", False),
            pytest.param(100, 100, "US", False),
            pytest.param(100, 100, None, False),
        ],
    )
    def test_is_mammogram(self, record, rows, columns, modality, exp):
        record = replace(record, Rows=rows, Columns=columns, Modality=modality, mammogram_type=None)
        assert record.is_mammogram == exp

    @pytest.mark.parametrize(
        "paddle,code,exp",
        [
            pytest.param("SPOT", None, True),
            pytest.param("SPOT COMPRESSION", None, True),
            pytest.param(None, None, False),
            pytest.param(None, "spot compression", True),
        ],
    )
    def test_is_spot_compression(self, record, paddle, code, exp):
        seq = Sequence([make_view_modifier_code(code)]) if code is not None else None
        record = replace(
            record,
            Modality="MG",
            PaddleDescription=paddle,
            ViewModifierCodeSequence=seq,
        )
        assert record.is_spot_compression == exp

    @pytest.mark.parametrize(
        "uid,exp",
        [
            pytest.param(SecondaryCaptureImageStorage, True),
            pytest.param(None, False),
            pytest.param("", False),
        ],
    )
    def test_is_secondary_capture(self, record, uid, exp):
        record = replace(
            record,
            SOPClassUID=uid,
        )
        assert record.is_secondary_capture == exp

    @pytest.mark.parametrize(
        "modality,exp",
        [
            pytest.param("MG", False),
            pytest.param("PR", True),
            pytest.param(None, False),
            pytest.param("", False),
        ],
    )
    def test_is_pr_file(self, record, modality, exp):
        record = replace(
            record,
            Modality=modality,
            mammogram_type=None,
        )
        assert record.is_pr_file == exp

    @pytest.mark.parametrize(
        "modality,exp",
        [
            pytest.param("MG", False),
            pytest.param("US", True),
            pytest.param(None, False),
            pytest.param("", False),
        ],
    )
    def test_is_ultrasound(self, record, modality, exp):
        record = replace(
            record,
            Modality=modality,
            mammogram_type=None,
        )
        assert record.is_ultrasound == exp

    @pytest.mark.parametrize(
        "modality,nf,mtype,exp",
        [
            pytest.param("MG", 1, MammogramType.FFDM, False),
            pytest.param("MG", 1, MammogramType.SFM, False),
            pytest.param("US", 2, None, False),
            pytest.param("MG", 2, MammogramType.TOMO, True),
            pytest.param("MG", 1, MammogramType.SVIEW, False),
        ],
    )
    def test_is_tomo(self, record, modality, nf, mtype, exp):
        record = replace(
            record,
            Modality=modality,
            mammogram_type=mtype,
            NumberOfFrames=nf,
        )
        assert record.is_tomo == exp

    @pytest.mark.parametrize(
        "modality,nf,mtype,exp",
        [
            pytest.param("MG", 1, MammogramType.FFDM, False),
            pytest.param("MG", 1, MammogramType.SFM, False),
            pytest.param("US", 2, None, False),
            pytest.param("MG", 2, MammogramType.TOMO, False),
            pytest.param("MG", 1, MammogramType.SVIEW, True),
        ],
    )
    def test_is_synthetic_view(self, record, modality, nf, mtype, exp):
        record = replace(
            record,
            Modality=modality,
            mammogram_type=mtype,
            NumberOfFrames=nf,
        )
        assert record.is_synthetic_view == exp

    @pytest.mark.parametrize(
        "modality,nf,mtype,exp",
        [
            pytest.param("MG", 1, MammogramType.FFDM, True),
            pytest.param("MG", 1, MammogramType.SFM, False),
            pytest.param("US", 2, None, False),
            pytest.param("MG", 2, MammogramType.TOMO, False),
            pytest.param("MG", 1, MammogramType.SVIEW, False),
        ],
    )
    def test_is_ffdm(self, record, modality, nf, mtype, exp):
        record = replace(
            record,
            Modality=modality,
            mammogram_type=mtype,
            NumberOfFrames=nf,
        )
        assert record.is_ffdm == exp

    @pytest.mark.parametrize(
        "modality,nf,mtype,exp",
        [
            pytest.param("MG", 1, MammogramType.FFDM, False),
            pytest.param("MG", 1, MammogramType.SFM, True),
            pytest.param("US", 2, None, False),
            pytest.param("MG", 2, MammogramType.TOMO, False),
            pytest.param("MG", 1, MammogramType.SVIEW, False),
        ],
    )
    def test_is_sfm(self, record, modality, nf, mtype, exp):
        record = replace(
            record,
            Modality=modality,
            mammogram_type=mtype,
            NumberOfFrames=nf,
        )
        assert record.is_sfm == exp

    @pytest.mark.parametrize(
        "view_pos,exp",
        [
            pytest.param(ViewPosition.MLO, True),
            pytest.param(ViewPosition.CC, True),
            *[pytest.param(x, False) for x in ViewPosition if x not in (ViewPosition.MLO, ViewPosition.CC)],
        ],
    )
    def test_is_standard_mammo_view(self, record, view_pos, exp):
        record = replace(record, view_position=view_pos)
        assert record.is_standard_mammo_view == exp

    def test_is_complete_mammo_case(self, record):
        # should be incomplete until after this loop
        records: List[FileRecord] = []
        for (laterality, view_pos) in STANDARD_MAMMO_VIEWS:
            assert not FileRecord.is_complete_mammo_case(records)
            rec = replace(record, laterality=laterality, view_position=view_pos)
            records.append(rec)
        assert FileRecord.is_complete_mammo_case(records)

    @pytest.mark.parametrize(
        "code,exp",
        [
            pytest.param("spot compression", False),
            pytest.param("magnified", True),
            pytest.param("magnification", True),
        ],
    )
    def test_is_magnified(self, record, code, exp):
        seq = Sequence([make_view_modifier_code(code)]) if code is not None else None
        record = replace(
            record,
            Modality="MG",
            ViewModifierCodeSequence=seq,
        )
        assert record.is_magnified == exp

    @pytest.mark.parametrize(
        "code,exp",
        [
            pytest.param("implant displaced", True),
            pytest.param("magnified", False),
            pytest.param("", False),
        ],
    )
    def test_is_implant_displaced(self, record, code, exp):
        seq = Sequence([make_view_modifier_code(code)]) if code is not None else None
        record = replace(
            record,
            Modality="MG",
            ViewModifierCodeSequence=seq,
        )
        assert record.is_implant_displaced == exp

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
    def test_year(self, record, study_date, exp):
        record = replace(
            record,
            StudyDate=study_date,
        )
        assert record.year == exp

    @pytest.mark.parametrize(
        "modality,mtype,spot,mag,id,laterality,view_pos,uid,exp",
        [
            pytest.param(
                "MG",
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
                "MG",
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
                "US",
                None,
                False,
                False,
                False,
                None,
                None,
                "1",
                "us_1.dcm",
            ),
            pytest.param(
                "MG",
                MammogramType.SVIEW,
                True,
                True,
                True,
                Laterality.RIGHT,
                ViewPosition.XCCL,
                "1",
                "synth_spot_mag_id_rxccl_1.dcm",
            ),
        ],
    )
    def test_standardized_filename(self, record, modality, mtype, spot, mag, id, laterality, view_pos, uid, exp):
        seq = []
        if spot:
            seq.append(make_view_modifier_code("spot compression"))
        if mag:
            seq.append(make_view_modifier_code("magnification"))
        if id:
            seq.append(make_view_modifier_code("implant displaced"))
        seq = Sequence(seq)
        record = replace(
            record,
            Modality=modality,
            mammogram_type=mtype,
            ViewModifierCodeSequence=seq,
            laterality=laterality,
            view_position=view_pos,
        )
        assert record.standardized_filename(uid) == Path(exp)

    @pytest.mark.parametrize(
        "series,sop,exp",
        [
            pytest.param("1.2", None, True),
            pytest.param(None, "1.2", True),
            pytest.param(None, None, False),
        ],
    )
    def test_has_image_uid(self, record, series, sop, exp):
        record = replace(record, SeriesInstanceUID=series, SOPInstanceUID=sop)
        assert record.has_image_uid == exp

    @pytest.mark.parametrize(
        "series,sop,prefer_sop,exp",
        [
            pytest.param("1.2", None, True, "1.2"),
            pytest.param("1.2", None, False, "1.2"),
            pytest.param(None, "1.2", True, "1.2"),
            pytest.param(None, "1.2", False, "1.2"),
            pytest.param("1.2", "2.3", True, "2.3"),
            pytest.param("1.2", "2.3", False, "1.2"),
            pytest.param(None, None, True, None, marks=pytest.mark.xfail(raises=AttributeError)),
            pytest.param(None, None, False, None, marks=pytest.mark.xfail(raises=AttributeError)),
        ],
    )
    def test_get_image_uid(self, record, series, sop, prefer_sop, exp):
        record = replace(record, SeriesInstanceUID=series, SOPInstanceUID=sop)
        uid = record.get_image_uid(prefer_sop=prefer_sop)
        assert uid == exp


@pytest.fixture
def dicom_files(tmp_path, dicom_file):
    paths = []
    for i in range(3):
        for j in range(3):
            dest = Path(tmp_path, f"subdir_{i}", f"file_{j}.dcm")
            dest.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(dicom_file, str(dest))
            paths.append(dest)
    return paths


@pytest.mark.parametrize("use_bar", [True, False])
@pytest.mark.parametrize("threads", [False, True])
@pytest.mark.parametrize("jobs", [None, 1, 2])
def test_record_iterator(dicom_files, use_bar, threads, jobs):
    records = record_iterator(dicom_files, jobs, use_bar, threads)
    assert set(rec.path for rec in records) == set(dicom_files)


class TestRecordCollection:
    @pytest.mark.parametrize("use_bar", [True, False])
    @pytest.mark.parametrize("threads", [False, True])
    @pytest.mark.parametrize("jobs", [None, 1, 2])
    def test_from_files(self, dicom_files, use_bar, threads, jobs):
        col = RecordCollection.from_files(dicom_files, jobs, use_bar, threads)
        assert set(col.path_lookup.keys()) == set(dicom_files)
        assert all(isinstance(v, FileRecord) for v in col.path_lookup.values())
        assert set(v.path for v in col.path_lookup.values()) == set(dicom_files)

    @pytest.mark.parametrize("use_bar", [True, False])
    @pytest.mark.parametrize("threads", [False, True])
    @pytest.mark.parametrize("jobs", [None, 1, 2])
    def test_from_dir(self, tmp_path, dicom_files, use_bar, threads, jobs):
        col = RecordCollection.from_dir(tmp_path, "*.dcm", jobs, use_bar, threads)
        assert set(col.path_lookup.keys()) == set(dicom_files)
        assert all(isinstance(v, FileRecord) for v in col.path_lookup.values())
        assert set(v.path for v in col.path_lookup.values()) == set(dicom_files)

    @pytest.fixture
    def collection(self, dicom_files):
        return RecordCollection.from_files(dicom_files)

    def test_len(self, collection, dicom_files):
        assert len(collection) == len(dicom_files)

    def test_study_lookup(self, collection, dicom_files):
        lookup = collection.study_lookup
        assert len(lookup) == 1
        records = lookup["1.3.6.1.4.1.5962.1.2.1.20040119072730.12322"]
        assert len(records) == len(dicom_files)
        assert all(isinstance(r, FileRecord) for r in records)

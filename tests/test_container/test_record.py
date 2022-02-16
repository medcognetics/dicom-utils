#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pydicom
import pytest

from dicom_utils.container import FileRecord, RecordCollection, record_iterator
from dicom_utils.types import SimpleImageType


class TestFileRecord:
    def test_create(self, dicom_file):
        rec = FileRecord.create(Path(dicom_file))
        dcm = pydicom.dcmread(dicom_file)
        assert rec.path == Path(dicom_file)
        assert rec.StudyInstanceUID == dcm.StudyInstanceUID
        assert rec.SeriesInstanceUID == dcm.SeriesInstanceUID
        assert rec.SOPInstanceUID == dcm.SOPInstanceUID
        assert rec.TransferSyntaxUID == dcm.file_meta.TransferSyntaxUID
        assert rec.Rows == dcm.Rows
        assert rec.Columns == dcm.Columns
        assert rec.NumberOfFrames == dcm.get("NumberOfFrames", None)
        assert rec.PhotometricInterpretation == dcm.PhotometricInterpretation
        assert rec.SimpleImageType == SimpleImageType.NORMAL
        assert rec.ManufacturerModelName == dcm.ManufacturerModelName
        assert rec.SeriesDescription == dcm.get("SeriesDescription", None)
        assert rec.PatientName == dcm.get("PatientName", None)
        assert rec.PatientID == dcm.get("PatientID", None)

    @pytest.fixture
    def record(self, dicom_file):
        return FileRecord.create(Path(dicom_file))

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

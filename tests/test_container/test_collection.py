#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import PathLike
from pathlib import Path
from typing import Optional, cast

import pydicom
import pytest

from dicom_utils.container import (
    FILTER_REGISTRY,
    HELPER_REGISTRY,
    SOPUID,
    DicomFileRecord,
    DicomImageFileRecord,
    FileRecord,
    MammogramFileRecord,
    RecordCollection,
    RecordCreator,
    RecordFilter,
    RecordHelper,
    StandardizedFilename,
    StudyUID,
    record_iterator,
)
from dicom_utils.container.collection import iterate_input_path
from dicom_utils.dicom_factory import DicomFactory


class TestIterateInputPath:
    def test_file(self, tmp_path):
        path = Path(tmp_path, "test.dcm")
        path.touch()
        result = list(iterate_input_path(path))
        assert result == [path]

    @pytest.mark.parametrize("max_depth", [None, 0, 1])
    def test_dir(self, tmp_path, max_depth):
        path = Path(tmp_path, "dir")
        path.mkdir()
        subfile = Path(path, "test.dcm")
        subfile.touch()
        result = list(iterate_input_path(path, max_depth))
        if max_depth == 0:
            assert result == [path]
        else:
            assert result == list(path.iterdir())

    @pytest.mark.parametrize("follow_text_files", [True, False])
    def test_text_file_of_files(self, tmp_path, follow_text_files):
        path = Path(tmp_path, "test.txt")
        subdir = Path(tmp_path, "dir")
        subdir.mkdir()
        with path.open("w") as f:
            targets = []
            for i in range(3):
                p = Path(subdir, f"test{i}.dcm")
                p.touch()
                targets.append(p)
                f.write(f"{p}\n")
        result = list(iterate_input_path(path, follow_text_files=follow_text_files))
        assert result == (targets if follow_text_files else [path])

    @pytest.mark.parametrize("max_depth", [None, 0, 1])
    def test_text_file_of_dirs(self, tmp_path, max_depth):
        path = Path(tmp_path, "test.txt")
        subdir = Path(tmp_path, "dir")
        subdir.mkdir()
        with path.open("w") as f:
            dir_targets = []
            file_targets = []
            for i in range(3):
                subsubdir = Path(subdir, f"subsubdir{i}")
                subsubdir.mkdir()
                p = Path(subsubdir, f"test{i}.txt")
                # write dummy file content to ensure we don't recursively read text files
                with p.open("w") as f2:
                    f2.write("nofile.txt")
                dir_targets.append(subsubdir)
                file_targets.append(p)
                f.write(f"{subsubdir}\n")
        result = list(iterate_input_path(path, max_depth))
        if max_depth == 0:
            assert result == dir_targets
        else:
            assert result == file_targets

    @pytest.mark.parametrize(
        "ignore_missing",
        [
            True,
            pytest.param(False, marks=pytest.mark.xfail(raises=FileNotFoundError)),
        ],
    )
    def test_missing_filepath(self, tmp_path, ignore_missing):
        path = Path(tmp_path, "test.txt")
        subdir = Path(tmp_path, "dir")
        subdir.mkdir()
        with path.open("w") as f:
            p = Path(subdir, "nofile.txt")
            f.write(f"{p}\n")
        result = list(iterate_input_path(path, ignore_missing=ignore_missing))
        assert result == []

    @pytest.mark.parametrize("follow_symlinks", [True, False])
    def test_handle_symlink_files(self, tmp_path, follow_symlinks):
        real_path = tmp_path / "real"
        symlink_path = tmp_path / "symlink"
        real_path.mkdir()
        symlink_path.mkdir()

        for i in range(3):
            p = real_path / f"test{i}.dcm"
            p.touch()
            symlink = symlink_path / f"test{i}.dcm"
            symlink.symlink_to(p)
            assert symlink.is_file()

        result = list(iterate_input_path(symlink_path, follow_symlinks=follow_symlinks))
        assert len(result) == (3 if follow_symlinks else 0)
        assert all(r.is_symlink() for r in result)

    @pytest.mark.parametrize("follow_symlinks", [True, False])
    def test_handle_symlink_dirs(self, tmp_path, follow_symlinks):
        real_path = tmp_path / "real"
        symlink_path = tmp_path / "symlink"
        real_path.mkdir()
        symlink_path.mkdir()

        for i in range(3):
            p = real_path / str(i) / f"test{i}.dcm"
            p.parent.mkdir()
            p.touch()
            symlink = symlink_path / str(i)
            symlink.symlink_to(p.parent)
            assert symlink.is_dir()

        result = list(iterate_input_path(symlink_path, follow_symlinks=follow_symlinks))
        assert len(result) == (3 if follow_symlinks else 0)
        assert all(r.parent.is_symlink() for r in result)


@pytest.fixture
def suffix():
    return ".dcm"


@pytest.fixture
def dicom_files(tmp_path, dicom_file, suffix):
    paths = []
    factory = DicomFactory(proto=dicom_file)
    for i in range(3):
        for j in range(3):
            study_uid = f"study_{i}"
            sop_uid = f"study_{i}_sop_{j}"
            dest = Path(tmp_path, f"subdir_{i}", f"file_{j}{suffix}")
            dest.parent.mkdir(exist_ok=True, parents=True)

            dcm = factory(StudyInstanceUID=study_uid, SOPInstanceUID=sop_uid)
            dcm.save_as(dest)
            paths.append(dest)
    return paths


class TestRecordCreator:
    @pytest.fixture
    def dicom_factory(self, tmp_path, dicom_file):
        def func(
            filename: PathLike = Path("foo.dcm"),
            Rows: Optional[int] = 128,
            Columns: Optional[int] = 128,
            Modality: Optional[str] = "MG",
            **kwargs,
        ):
            filename = Path(tmp_path, filename)
            filename.parent.mkdir(exist_ok=True, parents=True)

            with pydicom.dcmread(dicom_file) as dcm:
                dcm.Rows = Rows
                dcm.Columns = Columns
                dcm.Modality = Modality
                dcm.save_as(filename)
            return filename

        return func

    @pytest.mark.parametrize(
        "rows,columns,modality,exp",
        [
            pytest.param(128, 128, "MG", MammogramFileRecord),
            pytest.param(128, 128, "CT", DicomImageFileRecord),
            pytest.param(128, 128, "", DicomImageFileRecord),
            pytest.param(None, 128, "MG", DicomFileRecord),
            pytest.param(128, None, "CT", DicomFileRecord),
        ],
    )
    def test_create(self, rows, columns, modality, exp, dicom_factory):
        c = RecordCreator()
        rec = c(dicom_factory(Rows=rows, Columns=columns, Modality=modality))
        assert type(rec) == exp

    @pytest.mark.parametrize("suffix", [".txt", ".html", ".json"])
    def test_create_other_files(self, tmp_path, suffix):
        f = Path(tmp_path, "file").with_suffix(suffix)
        f.touch()
        c = RecordCreator()
        rec = c(f)
        assert type(rec) == FileRecord
        assert rec.path == f


class TestRecordIterator:
    @pytest.mark.parametrize("suffix", [".dcm", "", ".123"])
    @pytest.mark.parametrize("use_bar", [True, False])
    @pytest.mark.parametrize("threads", [False, True])
    @pytest.mark.parametrize("jobs", [None, 1, 2])
    def test_default(self, dicom_files, use_bar, threads, jobs):
        records = list(record_iterator(dicom_files, jobs, use_bar, threads, ignore_exceptions=False))
        assert all(isinstance(r, DicomFileRecord) for r in records)
        assert set(rec.path for rec in records) == set(dicom_files)

    def test_helpers(self, dicom_files):
        @HELPER_REGISTRY(name="dummy-pid")
        class PatientIDHelper(RecordHelper):
            def __call__(self, _, rec):
                return rec.replace(PatientID="TEST")

        records = list(record_iterator(dicom_files, helpers=["dummy-pid"], threads=True))
        assert all(isinstance(r, DicomFileRecord) for r in records)
        assert all(cast(DicomFileRecord, r).PatientID == "TEST" for r in records)

    def test_filters(self, dicom_files):
        @FILTER_REGISTRY(name="dummy-filter")
        class DummyFilter(RecordFilter):
            def path_is_valid(self, path: Path) -> bool:
                return str(path).endswith("0.dcm")

            def record_is_valid(self, rec: FileRecord) -> bool:
                return str(rec.path.parent).endswith("0")

        records = list(record_iterator(dicom_files, filters=["dummy-filter"], threads=True))
        assert all(isinstance(r, DicomFileRecord) for r in records)
        assert all(str(r.path).endswith("0.dcm") for r in records), "path_is_valid filter failed"
        assert all(str(r.path.parent).endswith("0") for r in records), "record_is_valid filter failed"


class TestRecordCollection:
    @pytest.mark.parametrize("use_bar", [True, False])
    @pytest.mark.parametrize("threads", [False, True])
    @pytest.mark.parametrize("jobs", [None, 1, 2])
    def test_from_files(self, dicom_files, use_bar, threads, jobs, caplog):
        col = RecordCollection.from_files(dicom_files, jobs, use_bar, threads)
        assert set(x.path for x in col) == set(dicom_files)
        assert all(isinstance(v, FileRecord) for v in col)
        assert set(v.path for v in col) == set(dicom_files)

    @pytest.mark.parametrize("use_bar", [True, False])
    @pytest.mark.parametrize("threads", [False, True])
    @pytest.mark.parametrize("jobs", [None, 1, 2])
    def test_from_dir(self, tmp_path, dicom_files, use_bar, threads, jobs):
        col = RecordCollection.from_dir(tmp_path, "*.dcm", jobs, use_bar, threads)
        assert set(x.path for x in col) == set(dicom_files)
        assert all(isinstance(v, FileRecord) for v in col)
        assert set(v.path for v in col) == set(dicom_files)

    @pytest.mark.parametrize("use_bar", [True, False])
    @pytest.mark.parametrize("threads", [False, True])
    @pytest.mark.parametrize("jobs", [None, 1, 2])
    def test_create(self, tmp_path, dicom_files, use_bar, threads, jobs):
        files = [tmp_path] + dicom_files
        col = RecordCollection.create(files, "*.dcm", jobs, use_bar, threads)
        assert set(x.path for x in col) == set(dicom_files)
        assert all(isinstance(v, FileRecord) for v in col)
        assert set(v.path for v in col) == set(dicom_files)

    @pytest.fixture
    def collection(self, dicom_files):
        return RecordCollection.from_files(dicom_files)

    def test_len(self, collection, dicom_files):
        assert len(collection) == len(dicom_files)

    def test_standardized_filenames(self, tmp_path, dicom_files):
        col = RecordCollection.from_dir(tmp_path, "*.dcm")
        pairs = list(col.standardized_filenames())
        names = [p[0] for p in pairs]
        assert all(isinstance(n, StandardizedFilename) for n in names)
        assert len(names) == len(col)
        assert len(set(names)) == len(names)

    def test_to_dict(self, collection, dicom_files):
        col_dict = collection.to_dict()
        assert isinstance(col_dict, dict)
        assert len(col_dict["records"]) == len(dicom_files)

    def test_unique(self):
        rec1 = DicomFileRecord(path=Path("foo.dcm"), SOPInstanceUID=SOPUID("123"), StudyInstanceUID=StudyUID("123"))
        rec2 = DicomFileRecord(path=Path("bar.dcm"), SOPInstanceUID=SOPUID("123"), StudyInstanceUID=StudyUID("123"))
        col = RecordCollection([rec1, rec2])
        assert len(col) == 1

    def test_repr(self, tmp_path, collection):
        result = repr(collection)
        expected_start = f"RecordCollection(length=9, types={{'DicomImageFileRecord': 9}}, parent={tmp_path}"
        assert result.startswith(expected_start)

    def test_repr_empty(self):
        result = repr(RecordCollection())
        expected_start = "RecordCollection(length=0, types={}, parent=None"
        assert result.startswith(expected_start)

    @pytest.mark.parametrize(
        "key",
        [
            lambda rec: rec.StudyInstanceUID,
            lambda rec: (rec.StudyInstanceUID, rec.SOPInstanceUID),
        ],
    )
    def test_group_by(self, collection, key):
        grouped = collection.group_by(key)
        assert isinstance(grouped, dict)
        assert all(isinstance(v, RecordCollection) for v in grouped.values())
        assert all(grouped.keys())

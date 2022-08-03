#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
from os import PathLike
from pathlib import Path
from typing import Optional, cast

import pydicom
import pytest

from dicom_utils.container import (
    FILTER_REGISTRY,
    HELPER_REGISTRY,
    ConcurrentMapper,
    DicomFileRecord,
    DicomImageFileRecord,
    FileRecord,
    MammogramFileRecord,
    RecordCollection,
    RecordCreator,
    RecordFilter,
    RecordHelper,
    record_iterator,
)


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


class TestConcurrentMapper:
    @pytest.mark.parametrize("jobs", [1, 4, None])
    @pytest.mark.parametrize("threads", [True, False])
    @pytest.mark.parametrize("length", [10, 1000])
    @pytest.mark.parametrize("chunksize", [1, 32])
    @pytest.mark.parametrize("inp_type", [list, set, iter])
    def test_map_list(self, threads, jobs, length, chunksize, inp_type):
        inp = list(range(length))
        exp = [str(x) for x in inp]
        inp = inp_type(inp)
        with ConcurrentMapper(threads=threads, jobs=jobs, chunksize=chunksize) as mapper:
            mapper.create_bar(desc="Processing", total=length)
            result = list(mapper(str, inp))
        assert result == exp

    @pytest.mark.parametrize("threads", [True, False])
    def test_forward_args(self, threads):
        inp = list(range(10))
        exp = [x + 2 for x in inp]
        with ConcurrentMapper(threads=threads, jobs=2, chunksize=2) as mapper:
            result = list(mapper(TestConcurrentMapper._add, inp, y=2))
        assert result == exp

    @staticmethod
    def _add(x: int, y: int):
        # NOTE: these must be pickleable methods
        return x + y

    @pytest.mark.parametrize(
        "ignore",
        [
            pytest.param(False, marks=pytest.mark.xfail(raises=ValueError)),
            True,
        ],
    )
    def test_ignore_exceptions(self, ignore):
        inp = list(range(1000))
        with ConcurrentMapper(jobs=2, chunksize=2, ignore_exceptions=ignore) as mapper:
            list(mapper(TestConcurrentMapper._raise, inp))

    @staticmethod
    def _raise(x):
        raise ValueError()

    @pytest.mark.parametrize("ignore", [False, True])
    def test_exception_callback(self, mocker, ignore):
        inp = list(range(1000))
        spy = mocker.spy(TestConcurrentMapper, "_callback")
        with ConcurrentMapper(jobs=2, chunksize=2, ignore_exceptions=ignore, exception_callback=spy) as mapper:
            try:
                list(mapper(TestConcurrentMapper._raise, inp))
            except ValueError:
                pass
            spy.assert_called()

    @staticmethod
    def _callback(f):
        pass

    @pytest.mark.parametrize("ex_type", [KeyboardInterrupt, SystemExit])
    def test_interrupt(self, ex_type):
        inp = list(range(4))
        with ConcurrentMapper(jobs=4, chunksize=2) as mapper:
            try:
                list(mapper(TestConcurrentMapper._loop, inp, ex_type=ex_type))
            except ex_type:
                pass

    @staticmethod
    def _loop(x, ex_type):
        if x == 0:
            raise ex_type()
        while True:
            pass


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
    def test_from_files(self, dicom_files, use_bar, threads, jobs):
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
        assert len(names) == 9
        assert len(set(names)) == len(names)

    def test_to_dict(self, collection, dicom_files):
        col_dict = collection.to_dict()
        assert isinstance(col_dict, dict)
        assert len(col_dict["records"]) == len(dicom_files)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import ClassVar, Type

import pytest

from dicom_utils.container import HELPER_REGISTRY, FileRecord, MammogramFileRecord, RecordHelper
from dicom_utils.container.collection import CollectionHelper
from dicom_utils.container.input import Input
from dicom_utils.container.output import ManifestOutput, Output, SymlinkFileOutput, is_2d_mammogram
from dicom_utils.dicom_factory import CompleteMammographyStudyFactory


@pytest.fixture
def dicom_files(tmp_path):
    paths = []
    for i in range(3):
        fact = CompleteMammographyStudyFactory(seed=i, PatientID=f"Patient{i}", StudyInstanceUID=f"Study{i}")
        for dcm in fact():
            path = (tmp_path / dcm.SOPInstanceUID).with_suffix(".dcm")
            dcm.save_as(path)
            paths.append(path)
    return paths


@HELPER_REGISTRY(name="set-implant")
class SetImplantHelper(RecordHelper):
    def __call__(self, path, record):
        if isinstance(record, MammogramFileRecord):
            record = record.replace(BreastImplantPresent="Y")
        return record


@HELPER_REGISTRY(name="add-new")
class AddNewRecord(CollectionHelper):
    def __call__(self, collection, *args, **kwargs):
        rec = FileRecord(Path("dummy.dcm"))
        collection.add(rec)
        return collection


def is_3d_case(col):
    return any(isinstance(rec, MammogramFileRecord) and rec.is_2d for rec in col)


class BaseOutputTest:
    INPUT_TYPE: ClassVar[Type[Output]]
    DEFAULT_LEN: ClassVar[int] = 3
    DEFAULT_IMAGES: ClassVar[int] = 12

    def test_basic(self, tmp_path, dicom_files):
        source = tmp_path
        dest = Path(tmp_path, "dest")
        dest.mkdir()
        inp = Input(source, use_bar=False)
        output = self.INPUT_TYPE(dest)
        result = output(inp)
        assert len(result) == self.DEFAULT_LEN
        for r in result.values():
            assert len(r) == self.DEFAULT_IMAGES  # type: ignore
            assert all(f.path.is_file() for f in r)

    def test_record_filter(self, tmp_path, dicom_files):
        source = tmp_path
        dest = Path(tmp_path, "dest")
        dest.mkdir()
        inp = Input(source, use_bar=False)
        output = self.INPUT_TYPE(dest, record_filter=is_2d_mammogram)
        result = output(inp)
        assert len(result) == self.DEFAULT_LEN
        for r in result.values():
            for write_result in r:
                assert all(not isinstance(rec, MammogramFileRecord) or rec.is_2d for rec in write_result.collection)

    def test_record_helper(self, tmp_path, dicom_files):
        source = tmp_path
        dest = Path(tmp_path, "dest")
        dest.mkdir()
        inp = Input(source, use_bar=False)
        output = self.INPUT_TYPE(dest, helpers=["set-implant"])
        result = output(inp)
        assert len(result) == self.DEFAULT_LEN
        for r in result.values():
            assert len(r) == self.DEFAULT_IMAGES  # type: ignore
            for write_result in r:
                assert all(not isinstance(rec, MammogramFileRecord) or rec.has_uid for rec in write_result.collection)


class TestSymlinkFileOutput(BaseOutputTest):
    INPUT_TYPE: ClassVar[Type[Output]] = SymlinkFileOutput

    def test_collection_filter(self, tmp_path, dicom_files):
        source = tmp_path
        dest = Path(tmp_path, "dest")
        dest.mkdir()
        inp = Input(source, use_bar=False)
        output = self.INPUT_TYPE(dest, record_filter=is_3d_case)
        result = output(inp)
        assert len(result) == 0

    def test_collection_filter_preserves_collection(self, tmp_path, dicom_files):
        source = tmp_path
        dest = Path(tmp_path, "dest")
        dest.mkdir()
        inp = Input(source, use_bar=False)
        output = self.INPUT_TYPE(dest, record_filter=is_3d_case)
        result = output(inp)
        assert len(result) == 0
        output = self.INPUT_TYPE(dest)
        result = output(inp)
        assert len(result) == self.DEFAULT_LEN

    def test_collection_helper(self, tmp_path, dicom_files):
        source = tmp_path
        dest = Path(tmp_path, "dest")
        dest.mkdir()
        inp = Input(source, use_bar=False)
        output = self.INPUT_TYPE(dest, helpers=["add-new"], threads=True)
        result = output(inp)
        assert len(result) == self.DEFAULT_LEN
        for r in result.values():
            assert len(r) == self.DEFAULT_IMAGES + 1  # type: ignore
            for write_result in r:
                assert any(rec.path.name == "dummy.dcm" for rec in write_result.collection)


class TestManifestOutput(BaseOutputTest):
    INPUT_TYPE: ClassVar[Type[Output]] = ManifestOutput
    DEFAULT_LEN: ClassVar[int] = 3
    DEFAULT_IMAGES: ClassVar[int] = 1

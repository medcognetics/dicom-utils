#!/usr/bin/env python
# -*- coding: utf-8 -*-
import runpy
import shutil
import sys
from pathlib import Path
from typing import List

import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence


@pytest.fixture
def dicom_image_file(tmpdir) -> str:
    filename = pydicom.data.get_testdata_file("CT_small.dcm")  # type: ignore
    shutil.copy(filename, tmpdir)
    return filename


def create_referenced_image_sequence(ref_ds: Dataset) -> Sequence:
    refd_image = Dataset()
    refd_image.ReferencedSOPClassUID = ref_ds.SOPClassUID
    refd_image.ReferencedSOPInstanceUID = ref_ds.SOPInstanceUID
    refd_image.ReferencedFrameNumber = "1"
    return Sequence([refd_image])


def create_graphic_object_sequence(data: List[float] = [0, 0, 1, 1], graphic_type: str = "CIRCLE") -> Sequence:
    graphic_object = Dataset()
    graphic_object.GraphicAnnotationUnits = "PIXEL"
    graphic_object.GraphicDimensions = 2
    graphic_object.GraphicData = data
    graphic_object.GraphicType = graphic_type
    return Sequence([graphic_object])


def create_graphic_annotation_sequence(ref_ds: Dataset) -> Sequence:
    graphic_annotation = Dataset()
    graphic_annotation.ReferencedImageSequence = create_referenced_image_sequence(ref_ds)
    graphic_annotation.GraphicObjectSequence = create_graphic_object_sequence()
    return Sequence([graphic_annotation])


@pytest.fixture
def dicom_annotation_file(tmpdir, dicom_image_file) -> None:
    ds = Dataset()
    ds.Modality = "PR"
    ds.GraphicAnnotationSequence = create_graphic_annotation_sequence(pydicom.dcmread(dicom_image_file))
    ds.file_meta = FileMetaDataset()
    ds.is_implicit_VR = False
    ds.is_little_endian = True
    ds.save_as(tmpdir / "annotation.dcm", write_like_original=False)


@pytest.mark.parametrize("out", [None, "foo.png"])
def test_dicom2img(dicom_image_file, dicom_annotation_file, tmp_path, out):
    sys.argv = [sys.argv[0], str(tmp_path), "--noblock"]

    if out is not None:
        path = Path(tmp_path, out)
        sys.argv.extend(["--output", str(path)])

    runpy.run_module("dicom_utils.cli.dicom2img", run_name="__main__", alter_sys=True)

    if out is not None:
        assert "path" in locals()
        assert locals()["path"].is_file()

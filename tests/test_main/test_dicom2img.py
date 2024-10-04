#!/usr/bin/env python
# -*- coding: utf-8 -*-
import runpy
import shutil
import sys
from pathlib import Path
from typing import List

import numpy as np
import pydicom
import pytest
from PIL import Image
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
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian  # type: ignore
    ds.save_as(tmpdir / "annotation.dcm", enforce_file_format=False)


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


@pytest.mark.parametrize(
    "out", [pytest.param(None, marks=pytest.mark.xfail(raises=NotImplementedError, strict=True)), "foo.gif"]
)
@pytest.mark.parametrize("handler", ["keep", "max-8-5"])
def test_dicom2img_3d(dicom_file_3d, tmp_path, out, handler):
    sys.argv = [sys.argv[0], str(tmp_path), "--noblock", "-v", handler]

    if out is not None:
        path = Path(tmp_path, out)
        sys.argv.extend(["--output", str(path)])

    runpy.run_module("dicom_utils.cli.dicom2img", run_name="__main__", alter_sys=True)

    if out is not None:
        assert "path" in locals()
        assert locals()["path"].is_file()


def test_bytes(dicom_image_file, dicom_annotation_file, tmp_path, capsysbinary):
    sys.argv = [sys.argv[0], str(tmp_path), "--noblock", "--bytes"]

    runpy.run_module("dicom_utils.cli.dicom2img", run_name="__main__", alter_sys=False)
    captured = capsysbinary.readouterr().out
    dest = Path(tmp_path, "out.png")
    with open(dest, "wb") as f:
        f.write(captured)

    img = np.asarray(Image.open(dest))
    assert img.shape == (128, 128, 3)

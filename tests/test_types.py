#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, cast

import pytest
from pydicom import DataElement

from dicom_utils.tags import Tag
from dicom_utils.types import ImageType, Laterality, MammogramType, PhotometricInterpretation, ViewPosition


@dataclass
class DummyElement:
    value: Any


def get_simple_image_type_test_cases():
    """Seen IMAGE_TYPE Fields:

    2D:
        ['ORIGINAL', 'PRIMARY']
        ['DERIVED', 'PRIMARY']
        ['ORIGINAL', 'PRIMARY', '', '', '', '', '', '', '150000']
        ['DERIVED', 'PRIMARY', 'POST_CONTRAST', 'SUBTRACTION', '', '', '', '', '50000']
        ['ORIGINAL', 'PRIMARY', 'POST_PROCESSED', '', '', '', '', '', '50000']
        ['DERIVED', 'PRIMARY', 'TOMO_PROJ', 'RIGHT', '', '', '', '', '150000'] (may be s-view, but no marker on image)
        ['DERIVED', 'SECONDARY']
        ['DERIVED', 'PRIMARY', 'TOMO_2D', 'LEFT', '', '', '', '', '150000']
        ['DERIVED', 'PRIMARY', 'TOMO_2D', 'RIGHT', '', '', '', '', '150000']

    S-View:
        ['DERIVED', 'PRIMARY', '', '', '', '', '', '', '150000']
        ['DERIVED', 'PRIMARY', 'TOMO', 'GENERATED_2D', '', '', '', '', '150000']
        ['DERIVED', 'PRIMARY', 'TOMOSYNTHESIS', 'GENERATED_2D', '', '', '', '', '150000']

    TOMO:
        ['DERIVED', 'PRIMARY', 'TOMOSYNTHESIS', 'NONE', '', '', '', '', '150000']

    """
    cases = []
    default: Dict[str, Any] = {"pixels": "ORIGINAL", "exam": "PRIMARY"}

    # 2D

    # ['ORIGINAL', 'PRIMARY']
    d = deepcopy(default)
    _ = pytest.param(d, MammogramType.FFDM, id="2d-1")
    cases.append(_)

    # ['DERIVED', 'PRIMARY']
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED"))
    _ = pytest.param(d, MammogramType.FFDM, id="2d-2")
    cases.append(_)

    # ['ORIGINAL', 'PRIMARY', '', '', '', '', '', '', '150000']
    d = deepcopy(default)
    d.update(dict(pixels="ORIGINAL", extras=["", "", "", "", "", "150000"]))
    _ = pytest.param(d, MammogramType.FFDM, id="2d-3")
    cases.append(_)

    # ['DERIVED', 'PRIMARY', 'POST_CONTRAST', 'SUBTRACTION', '', '', '', '', '50000']
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", flavor="POST_CONTRAST", extras=["SUBTRACTION", "", "", "50000"]))
    _ = pytest.param(d, MammogramType.FFDM, id="2d-4")
    cases.append(_)

    # ['ORIGINAL', 'PRIMARY', 'POST_PROCESSED', '', '', '', '', '', '50000']
    d = deepcopy(default)
    d.update(dict(pixels="ORIGINAL", flavor="POST_PROCESSED", extras=["", "", "50000"]))
    _ = pytest.param(d, MammogramType.FFDM, id="2d-5")
    cases.append(_)

    # ['DERIVED', 'PRIMARY', 'TOMO_PROJ', 'RIGHT', '', '', '', '', '150000'] (may be s-view, but no marker on image)
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", flavor="TOMO_PROJ", extras=["RIGHT", "", "50000"]))
    _ = pytest.param(d, MammogramType.FFDM, id="2d-6")
    cases.append(_)

    # ['DERIVED', 'SECONDARY']
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", exam="SECONDARY"))
    _ = pytest.param(d, MammogramType.FFDM, id="2d-7")
    cases.append(_)

    # ['DERIVED', 'PRIMARY', 'TOMO_2D', 'LEFT', '', '', '', '', '150000']
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", flavor="TOMO_2D", extras=["LEFT", "", "", "", "150000"]))
    _ = pytest.param(d, MammogramType.FFDM, id="2d-8")
    cases.append(_)

    # ['DERIVED', 'PRIMARY', 'TOMO_2D', 'RIGHT', '', '', '', '', '150000']
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", flavor="TOMO_2D", extras=["RIGHT", "", "", "", "150000"]))
    _ = pytest.param(d, MammogramType.FFDM, id="2d-9")
    cases.append(_)

    # S-VIEW

    # ['DERIVED', 'PRIMARY', 'TOMO', 'GENERATED_2D', '', '', '', '', '150000']
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", flavor="TOMO", extras=["GENERATED_2D", "", "", "", "150000"]))
    _ = pytest.param(d, MammogramType.SVIEW, id="sview-1")
    cases.append(_)

    # ['DERIVED', 'PRIMARY', 'TOMOSYNTHESIS', 'GENERATED_2D', '', '', '', '', '150000']
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", flavor="TOMOSYNTHESIS", extras=["GENERATED_2D", "", "", "", "150000"]))
    _ = pytest.param(d, MammogramType.SVIEW, id="sview-2")
    cases.append(_)

    # ['DERIVED', 'PRIMARY']
    # Data in SeriesDescription
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", series_description="L CC C-View"))
    _ = pytest.param(d, MammogramType.SVIEW, id="sview-3")
    cases.append(_)

    # ['DERIVED', 'PRIMARY']
    # Data in SeriesDescription
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", series_description="R MLO S-View"))
    _ = pytest.param(d, MammogramType.SVIEW, id="sview-4")
    cases.append(_)

    # TOMO

    # ['DERIVED', 'PRIMARY', 'TOMOSYNTHESIS', 'NONE', '', '', '', '', '150000']
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", NumberOfFrames=10, flavor="TOMOSYNTHESIS", extras=["NONE", "", "", "150000"]))
    _ = pytest.param(d, MammogramType.TOMO, id="tomo-1")
    cases.append(_)

    return cases


class TestImageType:
    def test_from_dicom(self, dicom_object):
        img_type = ImageType.from_dicom(dicom_object)
        assert img_type.pixels == "ORIGINAL"
        assert img_type.exam == "PRIMARY"
        assert img_type.flavor == "AXIAL"


class TestMammogramType:
    @pytest.mark.parametrize(
        "input_str, expected",
        [
            ("ffdm", MammogramType.FFDM),
            ("2d", MammogramType.FFDM),
            ("synth", MammogramType.SVIEW),
            ("s view", MammogramType.SVIEW),
            ("s-view", MammogramType.SVIEW),
            ("c-view", MammogramType.SVIEW),
            ("", MammogramType.UNKNOWN),
            ("unknown", MammogramType.UNKNOWN),
            ("tomo", MammogramType.TOMO),
            ("tomosynthesis", MammogramType.TOMO),
        ],
    )
    def test_from_str(self, input_str, expected):
        assert expected == MammogramType.from_str(input_str)


class TestPhotometricInterpretation:
    @pytest.mark.parametrize(
        "val,expected",
        [
            pytest.param(PhotometricInterpretation.UNKNOWN, False),
            pytest.param(PhotometricInterpretation.MONOCHROME1, True),
            pytest.param(PhotometricInterpretation.MONOCHROME2, True),
            pytest.param(PhotometricInterpretation.RGB, True),
        ],
    )
    def test_bool(self, val, expected):
        assert bool(val) == expected

    @pytest.mark.parametrize(
        "val,expected",
        [
            pytest.param("", PhotometricInterpretation.UNKNOWN),
            pytest.param("MONOCHROME1", PhotometricInterpretation.MONOCHROME1),
            pytest.param("MONOCHROME2", PhotometricInterpretation.MONOCHROME2),
            pytest.param("RGB", PhotometricInterpretation.RGB),
        ],
    )
    def test_from_str(self, val, expected):
        pm = PhotometricInterpretation.from_str(val)
        assert pm == expected

    @pytest.mark.parametrize(
        "val,expected",
        [
            pytest.param(PhotometricInterpretation.UNKNOWN, False),
            pytest.param(PhotometricInterpretation.MONOCHROME1, True),
            pytest.param(PhotometricInterpretation.MONOCHROME2, True),
            pytest.param(PhotometricInterpretation.RGB, False),
        ],
    )
    def test_is_monochrome(self, val, expected):
        assert val.is_monochrome == expected

    @pytest.mark.parametrize(
        "val,expected",
        [
            pytest.param(None, PhotometricInterpretation.UNKNOWN),
            pytest.param("", PhotometricInterpretation.UNKNOWN),
            pytest.param("MONOCHROME1", PhotometricInterpretation.MONOCHROME1),
            pytest.param("MONOCHROME2", PhotometricInterpretation.MONOCHROME2),
            pytest.param("RGB", PhotometricInterpretation.RGB),
        ],
    )
    def test_from_dicom(self, dicom_object, val, expected):
        if val is not None:
            de = DataElement(Tag.PhotometricInterpretation, "CS", val)
            dicom_object[Tag.PhotometricInterpretation] = de
        else:
            del dicom_object[Tag.PhotometricInterpretation]

        pm = PhotometricInterpretation.from_dicom(dicom_object)
        assert pm == expected


class TestLaterality:
    @pytest.mark.parametrize(
        "orient,expected",
        [
            (Laterality.RIGHT, False),
            (Laterality.LEFT, False),
            (Laterality.BILATERAL, False),
            (Laterality.UNKNOWN, True),
        ],
    )
    def test_is_unknown(self, orient, expected):
        assert orient.is_unknown == expected

    @pytest.mark.parametrize(
        "string,expected",
        [
            ("rcc", Laterality.RIGHT),
            ("lcc", Laterality.LEFT),
            ("rmlo", Laterality.RIGHT),
            ("lmlo", Laterality.LEFT),
            ("r-cc", Laterality.RIGHT),
            ("L-MLO", Laterality.LEFT),
            ("L CC", Laterality.LEFT),
            ("CCD", Laterality.RIGHT),
            ("CCE", Laterality.LEFT),
            ("MLOD", Laterality.RIGHT),
            ("MLOE", Laterality.LEFT),
            ("RML", Laterality.RIGHT),
            ("LML", Laterality.LEFT),
            ("foo", Laterality.UNKNOWN),
            ("bi", Laterality.BILATERAL),
            ("bilateral", Laterality.BILATERAL),
            ("", Laterality.UNKNOWN),
            ("none", Laterality.NONE),
        ],
    )
    def test_from_str(self, string, expected):
        orient = Laterality.from_str(string)
        assert orient == expected

    @pytest.mark.parametrize(
        "laterality,image_laterality,frame_laterality,expected",
        [
            ("L", None, None, Laterality.LEFT),
            (None, "L", None, Laterality.LEFT),
            (None, None, "L", Laterality.LEFT),
            ("L", "L", "L", Laterality.LEFT),
            ("R", None, None, Laterality.RIGHT),
            (None, "R", None, Laterality.RIGHT),
            (None, None, "R", Laterality.RIGHT),
            ("R", "R", "R", Laterality.RIGHT),
            (None, None, None, Laterality.UNKNOWN),
            ("", "", "", Laterality.UNKNOWN),
        ],
    )
    def test_from_tags(self, laterality, image_laterality, frame_laterality, expected):
        if frame_laterality:
            sfgs = [{Tag.FrameAnatomySequence: DummyElement([{Tag.FrameLaterality: DummyElement(frame_laterality)}])}]
        else:
            sfgs = None

        tags: Dict[int, Any] = {
            Tag.Laterality: laterality,
            Tag.ImageLaterality: image_laterality,
            Tag.SharedFunctionalGroupsSequence: sfgs,
        }
        tags = {k: v for k, v in tags.items() if v is not None}

        for k in tags:
            assert k in Laterality.get_required_tags()
        orient = Laterality.from_tags(tags)
        assert orient == expected

    def test_bool(self):
        expr = Laterality.UNKNOWN or Laterality.RIGHT or Laterality.UNKNOWN
        assert expr == Laterality.RIGHT

    def test_from_dicom(self, dicom_object):
        # trivial test since this wraps from_tags
        x = Laterality.from_dicom(dicom_object)
        assert x == Laterality.UNKNOWN


class TestViewPosition:
    @pytest.mark.parametrize(
        "orient,expected",
        [
            (ViewPosition.CC, False),
            (ViewPosition.CC, False),
            (ViewPosition.MLO, False),
            (ViewPosition.MLO, False),
            (ViewPosition.UNKNOWN, True),
        ],
    )
    def test_is_unknown(self, orient, expected):
        assert orient.is_unknown == expected

    @pytest.mark.parametrize(
        "string,expected",
        [
            ("rcc", ViewPosition.CC),
            ("lcc", ViewPosition.CC),
            ("rmlo", ViewPosition.MLO),
            ("lmlo", ViewPosition.MLO),
            ("r-cc", ViewPosition.CC),
            ("LMLO", ViewPosition.MLO),
            ("LCC", ViewPosition.CC),
            ("CCD", ViewPosition.CC),
            ("CCE", ViewPosition.CC),
            ("MLOD", ViewPosition.MLO),
            ("MLOE", ViewPosition.MLO),
            ("RML", ViewPosition.ML),
            ("LML", ViewPosition.ML),
            ("medio-lateral", ViewPosition.ML),
            ("medial-lateral", ViewPosition.ML),
            ("latero-medial", ViewPosition.LM),
            ("lateral-medial", ViewPosition.LM),
            ("cranio-caudal", ViewPosition.CC),
            ("caudal-cranial", ViewPosition.CC),
            ("medio-lateral oblique", ViewPosition.MLO),
            ("medial-lateral oblique", ViewPosition.MLO),
            ("latero-medial oblique", ViewPosition.LMO),
            ("lateral-medial oblique", ViewPosition.LMO),
            ("oblique medio-lateral", ViewPosition.MLO),
            ("oblique medial-lateral", ViewPosition.MLO),
            ("oblique latero-medial", ViewPosition.LMO),
            ("oblique lateral-medial", ViewPosition.LMO),
            ("cranio-caudal exaggerated laterally", ViewPosition.XCCL),
            ("cranio-caudal exaggerated medially", ViewPosition.XCCM),
            ("???", ViewPosition.UNKNOWN),
            ("foo", ViewPosition.UNKNOWN),
            ("", ViewPosition.UNKNOWN),
        ],
    )
    def test_from_str(self, string, expected):
        orient = ViewPosition.from_str(string)
        assert orient == expected

    @pytest.mark.parametrize(
        "val,expected",
        [
            ("MLO", ViewPosition.MLO),
            ("ML", ViewPosition.ML),
            ("CC", ViewPosition.CC),
            ("???", ViewPosition.UNKNOWN),
        ],
    )
    def test_from_tags(self, val, expected):
        tags = {0x00185101: val} if val is not None else {}
        for k in tags:
            assert k in ViewPosition.get_required_tags()
        orient = ViewPosition.from_tags(tags)
        assert orient == expected

    @pytest.mark.parametrize(
        "val,expected",
        [
            ("medio-lateral", ViewPosition.ML),
            ("medial-lateral", ViewPosition.ML),
            ("latero-medial", ViewPosition.LM),
            ("lateral-medial", ViewPosition.LM),
            ("cranio-caudal", ViewPosition.CC),
            ("caudal-cranial", ViewPosition.CC),
            ("medio-lateral oblique", ViewPosition.MLO),
            ("medial-lateral oblique", ViewPosition.MLO),
            ("latero-medial oblique", ViewPosition.LMO),
            ("lateral-medial oblique", ViewPosition.LMO),
            ("oblique medio-lateral", ViewPosition.MLO),
            ("oblique medial-lateral", ViewPosition.MLO),
            ("oblique latero-medial", ViewPosition.LMO),
            ("oblique lateral-medial", ViewPosition.LMO),
            ("cranio-caudal exaggerated laterally", ViewPosition.XCCL),
            ("cranio-caudal exaggerated medially", ViewPosition.XCCM),
            ("???", ViewPosition.UNKNOWN),
            ("", ViewPosition.UNKNOWN),
            (None, ViewPosition.UNKNOWN),
        ],
    )
    def test_from_view_code_sequence_tag(self, val, expected):
        view_code_sequence = [
            {
                "CodeMeaning": val,
            }
        ]
        assert Tag.ViewCodeSequence in ViewPosition.get_required_tags()
        orient = ViewPosition.from_view_code_sequence_tag(cast(DataElement, view_code_sequence))
        assert orient == expected

    def test_bool(self):
        expr = ViewPosition.UNKNOWN or ViewPosition.CC or ViewPosition.UNKNOWN
        assert expr == ViewPosition.CC

    def test_from_dicom(self, dicom_object):
        # trivial test since this wraps from_tags
        x = ViewPosition.from_dicom(dicom_object)
        assert x == ViewPosition.UNKNOWN

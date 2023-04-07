#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, cast

import pytest
from pydicom import DataElement
from pydicom.multival import MultiValue

from dicom_utils import DicomFactory
from dicom_utils.tags import Tag
from dicom_utils.types import (
    ImageType,
    Laterality,
    MammogramType,
    MammogramView,
    PhotometricInterpretation,
    PixelSpacing,
    ViewPosition,
)


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
    _ = pytest.param(d, MammogramType.SYNTH, id="sview-1")
    cases.append(_)

    # ['DERIVED', 'PRIMARY', 'TOMOSYNTHESIS', 'GENERATED_2D', '', '', '', '', '150000']
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", flavor="TOMOSYNTHESIS", extras=["GENERATED_2D", "", "", "", "150000"]))
    _ = pytest.param(d, MammogramType.SYNTH, id="sview-2")
    cases.append(_)

    # ['DERIVED', 'PRIMARY']
    # Data in SeriesDescription
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", series_description="L CC C-View"))
    _ = pytest.param(d, MammogramType.SYNTH, id="sview-3")
    cases.append(_)

    # ['DERIVED', 'PRIMARY']
    # Data in SeriesDescription
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", series_description="R MLO S-View"))
    _ = pytest.param(d, MammogramType.SYNTH, id="sview-4")
    cases.append(_)

    # TOMO

    # ['DERIVED', 'PRIMARY', 'TOMOSYNTHESIS', 'NONE', '', '', '', '', '150000']
    d = deepcopy(default)
    d.update(dict(pixels="DERIVED", NumberOfFrames=10, flavor="TOMOSYNTHESIS", extras=["NONE", "", "", "150000"]))
    _ = pytest.param(d, MammogramType.TOMO, id="tomo-1")
    cases.append(_)

    return cases


class TestImageType:
    @pytest.mark.parametrize(
        "pixels,exam,flavor,extras",
        [
            ("ORIGINAL", "PRIMARY", None, None),
            ("DERIVED", "PRIMARY", "AXIAL", None),
            ("DERIVED", "PRIMARY", "AXIAL", ["STEREO"]),
        ],
    )
    def test_from_dicom(self, dicom_object, pixels, exam, flavor, extras):
        dicom_object.ImageType = [x for x in (pixels, exam, flavor, *(extras or [])) if x]
        img_type = ImageType.from_dicom(dicom_object)
        assert img_type.pixels == pixels
        assert img_type.exam == exam
        assert img_type.flavor == flavor
        assert img_type.extras == extras

    @pytest.mark.parametrize(
        "pixels,exam,flavor,extras,target,expected",
        [
            ("ORIGINAL", "PRIMARY", "AXIAL", [], "ORIGINAL", True),
            ("ORIGINAL", "PRIMARY", "AXIAL", [], "PRIMARY", True),
            ("ORIGINAL", "PRIMARY", "AXIAL", [], "AXIAL", True),
            ("ORIGINAL", "PRIMARY", None, None, "AXIAL", False),
            ("ORIGINAL", "PRIMARY", None, ["AXIAL"], "AXIAL", True),
        ],
    )
    def test_contains(self, pixels, exam, flavor, extras, target, expected):
        img_type = ImageType(pixels, exam, flavor, extras)
        assert (target in img_type) == expected


class TestMammogramType:
    @pytest.mark.parametrize(
        "input_str, expected",
        [
            ("ffdm", MammogramType.FFDM),
            ("2d", MammogramType.FFDM),
            ("synth", MammogramType.SYNTH),
            ("s view", MammogramType.SYNTH),
            ("s-view", MammogramType.SYNTH),
            ("c-view", MammogramType.SYNTH),
            ("", MammogramType.UNKNOWN),
            ("unknown", MammogramType.UNKNOWN),
            ("tomo", MammogramType.TOMO),
            ("tomosynthesis", MammogramType.TOMO),
        ],
    )
    def test_from_str(self, input_str, expected):
        assert expected == MammogramType.from_str(input_str)

    @pytest.mark.parametrize(
        "val,exp",
        [
            (MammogramType.UNKNOWN, False),
            *[(val, True) for val in MammogramType if val != MammogramType.UNKNOWN],
        ],
    )
    def test_bool(self, val, exp):
        assert bool(val) == exp

    @pytest.mark.parametrize(
        "val,exp",
        [
            (MammogramType.UNKNOWN, True),
            *[(val, False) for val in MammogramType if val != MammogramType.UNKNOWN],
        ],
    )
    def test_is_unknown(self, val, exp):
        assert val.is_unknown == exp

    @pytest.mark.parametrize(
        "t1,t2,exp",
        [
            (MammogramType.FFDM, MammogramType.FFDM, False),
            (MammogramType.FFDM, MammogramType.SYNTH, True),
            (MammogramType.FFDM, MammogramType.TOMO, False),
            (MammogramType.FFDM, MammogramType.SFM, True),
            (MammogramType.SYNTH, MammogramType.FFDM, False),
            (MammogramType.SYNTH, MammogramType.SYNTH, False),
            (MammogramType.SYNTH, MammogramType.TOMO, False),
            (MammogramType.TOMO, MammogramType.FFDM, True),
            (MammogramType.TOMO, MammogramType.SYNTH, True),
            (MammogramType.TOMO, MammogramType.TOMO, False),
            (MammogramType.TOMO, MammogramType.SFM, True),
            (MammogramType.UNKNOWN, MammogramType.UNKNOWN, False),
            (MammogramType.FFDM, MammogramType.UNKNOWN, True),
            (MammogramType.SYNTH, MammogramType.UNKNOWN, True),
            (MammogramType.TOMO, MammogramType.UNKNOWN, True),
            (MammogramType.SFM, MammogramType.UNKNOWN, True),
        ],
    )
    def test_lt(self, t1, t2, exp):
        assert (t1 < t2) == exp
        # TODO: Move these to separate tests if we decide to change the ordering
        # convention in the future.
        assert t1.is_preferred_to(t2) == exp
        assert MammogramType.get_best([t1, t2]) == (t1 if exp else t2)

    @pytest.mark.parametrize(
        "t1,t2,exp",
        [
            (MammogramType.FFDM, MammogramType.FFDM, True),
            (MammogramType.FFDM, MammogramType.SYNTH, True),
            (MammogramType.FFDM, MammogramType.TOMO, False),
            (MammogramType.FFDM, MammogramType.SFM, True),
            (MammogramType.SYNTH, MammogramType.FFDM, False),
            (MammogramType.SYNTH, MammogramType.SYNTH, True),
            (MammogramType.SYNTH, MammogramType.TOMO, False),
            (MammogramType.TOMO, MammogramType.FFDM, True),
            (MammogramType.TOMO, MammogramType.SYNTH, True),
            (MammogramType.TOMO, MammogramType.TOMO, True),
            (MammogramType.TOMO, MammogramType.SFM, True),
            (MammogramType.UNKNOWN, MammogramType.UNKNOWN, True),
            (MammogramType.FFDM, MammogramType.UNKNOWN, True),
            (MammogramType.SYNTH, MammogramType.UNKNOWN, True),
            (MammogramType.TOMO, MammogramType.UNKNOWN, True),
            (MammogramType.SFM, MammogramType.UNKNOWN, True),
        ],
    )
    def test_le(self, t1, t2, exp):
        assert (t1 <= t2) == exp

    @pytest.mark.parametrize(
        "t1,t2,exp",
        [
            (MammogramType.FFDM, MammogramType.FFDM, False),
            (MammogramType.FFDM, MammogramType.SYNTH, False),
            (MammogramType.FFDM, MammogramType.TOMO, True),
            (MammogramType.FFDM, MammogramType.SFM, False),
            (MammogramType.SYNTH, MammogramType.FFDM, True),
            (MammogramType.SYNTH, MammogramType.SYNTH, False),
            (MammogramType.SYNTH, MammogramType.TOMO, True),
            (MammogramType.TOMO, MammogramType.FFDM, False),
            (MammogramType.TOMO, MammogramType.SYNTH, False),
            (MammogramType.TOMO, MammogramType.TOMO, False),
            (MammogramType.TOMO, MammogramType.SFM, False),
            (MammogramType.UNKNOWN, MammogramType.UNKNOWN, False),
            (MammogramType.FFDM, MammogramType.UNKNOWN, False),
            (MammogramType.SYNTH, MammogramType.UNKNOWN, False),
            (MammogramType.TOMO, MammogramType.UNKNOWN, False),
            (MammogramType.SFM, MammogramType.UNKNOWN, False),
            (MammogramType.UNKNOWN, MammogramType.FFDM, True),
        ],
    )
    def test_gt(self, t1, t2, exp):
        assert (t1 > t2) == exp

    @pytest.mark.parametrize(
        "t1,t2,exp",
        [
            (MammogramType.FFDM, MammogramType.FFDM, True),
            (MammogramType.FFDM, MammogramType.SYNTH, False),
            (MammogramType.FFDM, MammogramType.TOMO, True),
            (MammogramType.FFDM, MammogramType.SFM, False),
            (MammogramType.SYNTH, MammogramType.FFDM, True),
            (MammogramType.SYNTH, MammogramType.SYNTH, True),
            (MammogramType.SYNTH, MammogramType.TOMO, True),
            (MammogramType.TOMO, MammogramType.FFDM, False),
            (MammogramType.TOMO, MammogramType.SYNTH, False),
            (MammogramType.TOMO, MammogramType.TOMO, True),
            (MammogramType.TOMO, MammogramType.SFM, False),
            (MammogramType.UNKNOWN, MammogramType.UNKNOWN, True),
            (MammogramType.FFDM, MammogramType.UNKNOWN, False),
            (MammogramType.SYNTH, MammogramType.UNKNOWN, False),
            (MammogramType.TOMO, MammogramType.UNKNOWN, False),
            (MammogramType.SFM, MammogramType.UNKNOWN, False),
            (MammogramType.UNKNOWN, MammogramType.FFDM, True),
        ],
    )
    def test_ge(self, t1, t2, exp):
        assert (t1 >= t2) == exp


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
            (Laterality.NONE, False),
        ],
    )
    def test_is_unknown(self, orient, expected):
        assert orient.is_unknown == expected

    @pytest.mark.parametrize(
        "orient,expected",
        [
            (Laterality.RIGHT, False),
            (Laterality.LEFT, False),
            (Laterality.BILATERAL, False),
            (Laterality.UNKNOWN, True),
            (Laterality.NONE, True),
        ],
    )
    def test_is_unknown_or_none(self, orient, expected):
        assert orient.is_unknown_or_none == expected

    @pytest.mark.parametrize(
        "orient,expected",
        [
            (Laterality.RIGHT, True),
            (Laterality.LEFT, True),
            (Laterality.BILATERAL, False),
            (Laterality.UNKNOWN, False),
            (Laterality.NONE, False),
        ],
    )
    def test_is_unilateral(self, orient, expected):
        assert orient.is_unilateral == expected

    @pytest.mark.parametrize(
        "l1,l2,expected",
        [
            (Laterality.RIGHT, Laterality.BILATERAL, Laterality.BILATERAL),
            (Laterality.LEFT, Laterality.RIGHT, Laterality.BILATERAL),
            (Laterality.RIGHT, Laterality.LEFT, Laterality.BILATERAL),
            (Laterality.LEFT, Laterality.LEFT, Laterality.LEFT),
            (Laterality.RIGHT, Laterality.NONE, Laterality.RIGHT),
            (Laterality.NONE, Laterality.NONE, Laterality.NONE),
            (Laterality.UNKNOWN, Laterality.NONE, Laterality.NONE),
            (Laterality.BILATERAL, Laterality.UNKNOWN, Laterality.BILATERAL),
        ],
    )
    def test_reduce(self, l1, l2, expected):
        assert l1.reduce(l2) == expected

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

    @pytest.mark.parametrize(
        "value,expected",
        [
            (["A", "FL"], Laterality.RIGHT),
            (["A", "FR"], Laterality.LEFT),
            (["P", "FL"], Laterality.RIGHT),
            (["P", "FR"], Laterality.LEFT),
            (["A", "L"], Laterality.RIGHT),
            (["A", "R"], Laterality.LEFT),
            (["P", "L"], Laterality.RIGHT),
            (["P", "R"], Laterality.LEFT),
            (["P"], Laterality.UNKNOWN),
            ([], Laterality.UNKNOWN),
        ],
    )
    def test_from_patient_orientation(self, value, expected):
        orient = Laterality.from_patient_orientation(value)
        assert orient == expected

    def test_bool(self):
        expr = Laterality.UNKNOWN or Laterality.RIGHT or Laterality.UNKNOWN
        assert expr == Laterality.RIGHT

    def test_from_dicom(self, dicom_object):
        # trivial test since this wraps from_tags
        x = Laterality.from_dicom(dicom_object)
        assert x == Laterality.UNKNOWN

    @pytest.mark.parametrize(
        "lat,expected",
        [
            (Laterality.RIGHT, Laterality.LEFT),
            (Laterality.LEFT, Laterality.RIGHT),
            (Laterality.BILATERAL, Laterality.UNKNOWN),
            (Laterality.UNKNOWN, Laterality.UNKNOWN),
            (Laterality.NONE, Laterality.UNKNOWN),
        ],
    )
    def test_opposite(self, lat, expected):
        assert lat.opposite == expected


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
            ("axillary tail", ViewPosition.AT),
            ("???", ViewPosition.UNKNOWN),
            ("foo", ViewPosition.UNKNOWN),
            ("", ViewPosition.UNKNOWN),
            ("rxccl", ViewPosition.XCCL),
            ("lxccl", ViewPosition.XCCL),
            ("rxccm", ViewPosition.XCCM),
            ("lxccm", ViewPosition.XCCM),
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
            ("axillary tail", ViewPosition.AT),
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

    @pytest.mark.parametrize(
        "value,expected",
        [
            (["A", "FL"], ViewPosition.MLO),
            (["A", "FR"], ViewPosition.MLO),
            (["P", "FL"], ViewPosition.MLO),
            (["P", "FR"], ViewPosition.MLO),
            (["A", "L"], ViewPosition.CC),
            (["A", "R"], ViewPosition.CC),
            (["P", "L"], ViewPosition.CC),
            (["P", "R"], ViewPosition.CC),
            (["P"], ViewPosition.UNKNOWN),
            ("CC", ViewPosition.CC),
            ("MLO", ViewPosition.MLO),
            ([], ViewPosition.UNKNOWN),
        ],
    )
    def test_from_patient_orientation(self, value, expected):
        orient = ViewPosition.from_patient_orientation(value)
        assert orient == expected

    def test_bool(self):
        expr = ViewPosition.UNKNOWN or ViewPosition.CC or ViewPosition.UNKNOWN
        assert expr == ViewPosition.CC

    @pytest.mark.parametrize(
        "view_pos,code,modifier_code,exp",
        [
            pytest.param("MLO", None, None, ViewPosition.MLO),
            pytest.param(None, "cranio-caudal", None, ViewPosition.CC),
            pytest.param(None, "medio-lateral oblique", "axillary tail", ViewPosition.AT),
        ],
    )
    def test_from_dicom(self, view_pos, code, modifier_code, exp):
        factory = DicomFactory(Modality="MG")
        overrides = {}
        if view_pos is not None:
            overrides["ViewPosition"] = view_pos
        if code is not None:
            overrides["ViewCodeSequence"] = DicomFactory.code_sequence(code)
        if modifier_code is not None:
            overrides["ViewModifierCodeSequence"] = DicomFactory.code_sequence(modifier_code)

        dcm = factory(**overrides)
        x = ViewPosition.from_dicom(dcm)
        assert x == exp

    @pytest.mark.parametrize(
        "value,expected",
        [
            (ViewPosition.MLO, True),
            (ViewPosition.CC, True),
            *[(view, False) for view in ViewPosition if view not in (ViewPosition.MLO, ViewPosition.CC)],
        ],
    )
    def test_is_standard_view(self, value, expected):
        assert value.is_standard_view == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            (view, view in (ViewPosition.MLO, ViewPosition.ML, ViewPosition.LM, ViewPosition.LMO))
            for view in ViewPosition
        ],
    )
    def test_is_mlo_like(self, value, expected):
        assert value.is_mlo_like == expected

    @pytest.mark.parametrize(
        "value,expected",
        [(view, view in (ViewPosition.CC, ViewPosition.XCCL, ViewPosition.XCCM)) for view in ViewPosition],
    )
    def test_is_cc_like(self, value, expected):
        assert value.is_cc_like == expected


class TestPixelSpacing:
    @pytest.mark.parametrize(
        "string,exp",
        [
            pytest.param("[0.01, 0.01]", PixelSpacing(0.01, 0.01)),
            pytest.param("[0.001, 0.01]", PixelSpacing(0.001, 0.01)),
            pytest.param("[1.0e-002, 2.0e-002]", PixelSpacing(0.01, 0.02)),
            pytest.param("[0.001-0.01]", PixelSpacing(0.001, 0.01)),
            pytest.param("[0.001;0.01]", PixelSpacing(0.001, 0.01)),
            pytest.param("[100, 100]", PixelSpacing(100, 100)),
            pytest.param("[001e002, 001e002]", PixelSpacing(100, 100)),
            pytest.param("", None, marks=pytest.mark.xfail(raises=ValueError)),
            pytest.param("[foo, bar]", None, marks=pytest.mark.xfail(raises=ValueError)),
        ],
    )
    def test_from_str(self, string, exp):
        actual = PixelSpacing.from_str(string)
        assert actual == exp

    @pytest.mark.parametrize(
        "tag,value,exp",
        [
            pytest.param(Tag.PixelSpacing, MultiValue(str, [0.01, 0.02]), PixelSpacing(0.01, 0.02)),
            pytest.param(Tag.ImagerPixelSpacing, MultiValue(str, [0.01, 0.02]), PixelSpacing(0.01, 0.02)),
        ],
    )
    def test_from_dicom_pixel_spacing(self, tag, value, exp):
        factory = DicomFactory(**{tag.name: value})
        dcm = factory()
        actual = PixelSpacing.from_dicom(dcm)
        assert actual == exp


class TestMammogramView:
    @pytest.mark.parametrize(
        "lat_in,view_in,lat_out,view_out",
        [
            (None, None, Laterality.UNKNOWN, ViewPosition.UNKNOWN),
            (Laterality.LEFT, None, Laterality.LEFT, ViewPosition.UNKNOWN),
            (None, ViewPosition.MLO, Laterality.UNKNOWN, ViewPosition.MLO),
        ],
    )
    def test_create(self, lat_in, view_in, lat_out, view_out):
        assert MammogramView.create(lat_in, view_in) == MammogramView(lat_out, view_out)

    @pytest.mark.parametrize(
        "lat_in,view_in,lat_out,view_out",
        [
            ("L", "MLO", Laterality.LEFT, ViewPosition.MLO),
            ("", "MLO", Laterality.UNKNOWN, ViewPosition.MLO),
            ("L", "", Laterality.LEFT, ViewPosition.UNKNOWN),
            ("R", "MLO", Laterality.RIGHT, ViewPosition.MLO),
        ],
    )
    def test_from_dicom(self, lat_in, view_in, lat_out, view_out):
        fact = DicomFactory(ViewPosition=view_in, Laterality=lat_in)
        dcm = fact()
        assert MammogramView.from_dicom(dcm) == MammogramView(lat_out, view_out)

    @pytest.mark.parametrize(
        "value,expected",
        [
            (view, view in (ViewPosition.MLO, ViewPosition.ML, ViewPosition.LM, ViewPosition.LMO))
            for view in ViewPosition
        ],
    )
    def test_is_mlo_like(self, value, expected):
        mview = MammogramView(view=value)
        assert mview.is_mlo_like == expected

    @pytest.mark.parametrize(
        "value,expected",
        [(view, view in (ViewPosition.CC, ViewPosition.XCCL, ViewPosition.XCCM)) for view in ViewPosition],
    )
    def test_is_cc_like(self, value, expected):
        mview = MammogramView(view=value)
        assert mview.is_cc_like == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            (ViewPosition.MLO, True),
            (ViewPosition.CC, True),
            *[(view, False) for view in ViewPosition if view not in (ViewPosition.MLO, ViewPosition.CC)],
        ],
    )
    def test_is_standard_view(self, value, expected):
        mview = MammogramView(view=value)
        assert mview.is_standard_mammo_view == expected

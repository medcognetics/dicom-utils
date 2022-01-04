#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from pydicom.tag import Tag as PydicomTag

from dicom_utils.tags import PHITags, Tag, create_tag, get_display_width, is_phi


class TestTag:
    @pytest.mark.parametrize(
        "tag,expected",
        [
            pytest.param(Tag.SeriesInstanceUID, 0x0020000E),
            pytest.param(Tag.StudyInstanceUID, 0x0020000D),
        ],
    )
    def test_values(self, tag, expected):
        assert tag == expected

    def test_repr(self):
        t = Tag.PatientAge
        assert isinstance(str(t), str)
        assert isinstance(repr(t), str)

    @pytest.mark.parametrize(
        "tag,expected",
        [
            pytest.param(Tag.SeriesInstanceUID, 0x0020),
            pytest.param(Tag.StudyInstanceUID, 0x0020),
            pytest.param(Tag.EthnicGroup, 0x0010),
        ],
    )
    def test_group(self, tag, expected):
        assert tag.group == expected

    @pytest.mark.parametrize(
        "tag,expected",
        [
            pytest.param(Tag.SeriesInstanceUID, 0x000E),
            pytest.param(Tag.StudyInstanceUID, 0x000D),
        ],
    )
    def test_element(self, tag, expected):
        assert tag.element == expected


def test_num_phi_tags():
    assert len(PHITags) == 239


@pytest.mark.parametrize(
    "tag,phi",
    [
        pytest.param(Tag.StudyInstanceUID, False),
        pytest.param(Tag.SeriesInstanceUID, False),
        pytest.param(Tag.SOPInstanceUID, False),
        pytest.param(Tag.EthnicGroup, False),
        pytest.param(Tag.PatientAge, False),
        pytest.param(Tag.InstitutionName, False),
        pytest.param(Tag.InstitutionAddress, False),
        pytest.param(Tag.PatientBirthDate, True),
        pytest.param(Tag.PatientName, True),
        pytest.param(Tag.Occupation, True),
    ],
)
def test_is_phi(tag, phi):
    assert is_phi(tag) == phi


@pytest.mark.parametrize(
    "val,tag",
    [
        pytest.param("StudyInstanceUID", Tag.StudyInstanceUID),
        pytest.param("SOPInstanceUID", Tag.SOPInstanceUID),
        pytest.param("EthnicGroup", Tag.EthnicGroup),
        pytest.param("PatientAge", Tag.PatientAge),
        pytest.param(Tag.PatientAge, Tag.PatientAge),
        pytest.param(PydicomTag("PatientAge"), Tag.PatientAge),
        pytest.param("FooBar", None, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(1.0, None, marks=pytest.mark.xfail(raises=TypeError)),
    ],
)
def test_create_tag(val, tag):
    assert create_tag(val) == tag


def test_get_display_width():
    tags = [
        Tag.StudyInstanceUID,
        Tag.SeriesInstanceUID,
        Tag.PatientAge,
    ]
    expected = max(len(str(t)) for t in tags)
    assert get_display_width(tags) == expected
    assert get_display_width([]) == 0

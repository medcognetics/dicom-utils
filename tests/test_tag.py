#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest

from dicom_utils.tags import PHITags, Tag, is_phi


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

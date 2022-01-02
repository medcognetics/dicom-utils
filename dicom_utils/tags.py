#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Final, ClassVar, Iterable
from enum import IntEnum

from ._tag_enum import Tag


# Add PHI tags based on keyword
PHITags: Final = {Tag.MedicalRecordLocator, Tag.PatientName, Tag.Occupation}
keywords = ["address", "physician", "phone", "birth", "date", "ident"]
for t in Tag:
    name = t.name.lower()
    if any(keyword in name for keyword in keywords):
        PHITags.add(t)

# Remove certain tags picked up by keyword matching
PHITags.remove(Tag.PatientIdentityRemoved)
PHITags.remove(Tag.DeidentificationMethod)
PHITags.remove(Tag.DeidentificationMethodCodeSequence)
PHITags.remove(Tag.DateOfLastDetectorCalibration)
PHITags.remove(Tag.InstitutionAddress)


def is_phi(tag: Tag) -> bool:
    r"""Checks if a tag is known protected health information (PHI)"""
    return tag in PHITags


def tag_from_string(s: str) -> Tag:
    r"""Create a tag from a string keyword"""
    try:
        return getattr(Tag, s)
    except AttributeError:
        raise ValueError(f"Invalid tag {s}")


def get_display_width(tags: Iterable[Tag]) -> int:
    r"""Returns the width of the longest tag repr"""
    return max(len(str(tag)) for tag in tags)


ADDRESS: Final = int("0x" + "".join(f"{ord(v):2x}" for v in "MC"), 16)

def make_full_tag(x: int) -> int:
    return (ADDRESS << 16) + x

class MedcogTag(IntEnum):
    # descriptors
    Hash = 0
    BiopsyProven = 1
    AmericanData = 1
    LastModifiedDate = 1
    SyntheticImage = 1

    # patient descriptors
    Nationality = 1
    PostSurgical = 2
    PatientHistoryOfCancer = 1
    FamilyHistoryOfCancer = 1

    # global findings - study level
    StudyIsAbnormal = 2
    StudyIsActionable = 2
    StudyIsMalignant = 2
    StudyLesionTypes = 2
    StudyPathologyType = 2
    StudyBirads = 3

    # global findings - series level
    SeriesIsAbnormal = 2
    SeriesIsActionable = 2
    SeriesIsMalignant = 2
    SeriesLesionTypes = 2
    SeriesPathologyType = 2
    SeriesBirads = 2

    # regional findings
    RegionalAnnType = 1
    ROITypes = 1
    ROITraits = 1

    # FDA ground truthing assessments
    GroundTrutherNames = 5
    GroundTrutherAssessment = 5
    GroundTrutherNotes = 5
    GroundTrutherDensity = 5
    GroundTrutherReadTime = 5

    @property
    def group(self) -> int:
        return self >> 16

    @property
    def element(self) -> int:
        return self & 0xFFFF

    def __str__(self) -> str:
        tag_repr = "<{0:04x},{1:04x}>".format(self.group, self.element)
        return f"{tag_repr} {self.name}"

    def __repr__(self) -> str:
        return str(self)

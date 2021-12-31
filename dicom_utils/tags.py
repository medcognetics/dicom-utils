#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Final

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

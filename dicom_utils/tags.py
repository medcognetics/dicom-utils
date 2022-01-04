#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Final, Iterable, Union

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


def create_tag(val: Union[str, int]) -> Tag:
    r"""Create a tag from a string keyword or int value"""
    if isinstance(val, str):
        try:
            return getattr(Tag, val)
        except AttributeError:
            raise ValueError(f"Invalid tag {val}")
    elif isinstance(val, int):
        return Tag(val)
    else:
        raise TypeError(f"Expected int or str, found {type(val)}")


def get_display_width(tags: Iterable[Tag]) -> int:
    r"""Returns the width of the longest tag string"""
    return max((len(str(tag)) for tag in tags), default=0)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Final, Iterable, Tuple, Union

from dicomanonymizer.dicomfields import ALL_TAGS
from pydicom.tag import Tag as PydicomTag

from ._tag_enum import Tag


def create_tag(val: Union[str, int, Tuple[int, int]]) -> Tag:
    r"""Create a tag from a string keyword or int value"""
    if isinstance(val, str):
        try:
            return getattr(Tag, val)
        except AttributeError:
            raise ValueError(f"Invalid tag {val}")
    elif isinstance(val, int):
        return Tag(val)
    elif isinstance(val, tuple):
        return Tag(PydicomTag(*val))
    else:
        raise TypeError(f"Expected int, str or 2-tuple - found {type(val)}")


def get_display_width(tags: Iterable[Tag]) -> int:
    r"""Returns the width of the longest tag string"""
    return max((len(str(tag)) for tag in tags), default=0)


# Add PHI tags based on keyword
PHITags: Final = {create_tag(t) for t in ALL_TAGS if len(t) == 2}


# Remove certain tags picked up by keyword matching
# PHITags.remove(Tag.PatientIdentityRemoved)
# PHITags.remove(Tag.DeidentificationMethod)
# PHITags.remove(Tag.DeidentificationMethodCodeSequence)
# PHITags.remove(Tag.DateOfLastDetectorCalibration)
# PHITags.remove(Tag.InstitutionAddress)


def is_phi(tag: Tag) -> bool:
    r"""Checks if a tag is known protected health information (PHI)"""
    return tag in PHITags

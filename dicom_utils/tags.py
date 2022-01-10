#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Iterable, Tuple, Union

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

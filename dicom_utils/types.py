#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum

from pydicom.dataset import FileDataset


Dicom = FileDataset


class ImageType(Enum):
    UNKNOWN = 0
    NORMAL = 1
    SVIEW = 2
    TOMO = 3

    def __str__(self) -> str:
        if self is ImageType.UNKNOWN:
            return "unknown"
        elif self is ImageType.NORMAL:
            return "2d"
        elif self is ImageType.SVIEW:
            return "s-view"
        elif self is ImageType.TOMO:
            return "tomo"
        else:
            raise RuntimeError("unknown ImageType value")


__all__ = ["Dicom", "ImageType"]

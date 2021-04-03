#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    pass
except ImportError:
    print("DICOM operations require pydicom package")
    raise

try:
    from .version import __version__
except ImportError:
    __version__ = "Unknown"

from .dicom import NoImageError, read_image
from .metadata import add_patient_age, dicom_to_json, drop_fields_by_length, get_date


__all__ = [
    "__version__",
    "add_patient_age",
    "dicom_to_json",
    "drop_empty_tags",
    "drop_fields_by_length",
    "get_date",
    "NoImageError",
    "read_image",
]

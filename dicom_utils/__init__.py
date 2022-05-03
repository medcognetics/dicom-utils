#!/usr/bin/env python
# -*- coding: utf-8 -*-

# avoid BrokenPipeError, KeyboardInterrupt
from .anonymize import anonymize
from .dicom import NoImageError, read_dicom_image
from .metadata import add_patient_age, dicom_to_json, drop_fields_by_length, get_date
from .volume import KeepVolume, SliceAtLocation, UniformSample, VolumeHandler
from warnings import filterwarnings


try:
    pass
except ImportError:
    print("DICOM operations require pydicom package")
    raise

try:
    from .version import __version__
except ImportError:
    __version__ = "Unknown"

filterwarnings("ignore", ".*Invalid value for VR UI.*")


__all__ = [
    "__version__",
    "add_patient_age",
    "anonymize",
    "dicom_to_json",
    "drop_empty_tags",
    "drop_fields_by_length",
    "get_date",
    "NoImageError",
    "read_dicom_image",
    "VolumeHandler",
    "KeepVolume",
    "SliceAtLocation",
    "UniformSample",
]

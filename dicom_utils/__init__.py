#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib.metadata

# avoid BrokenPipeError, KeyboardInterrupt
import os
import warnings
from typing import Final, Iterable, Literal, cast

from .anonymize import anonymize
from .dicom import NoImageError, read_dicom_image
from .dicom_factory import DicomFactory
from .metadata import add_patient_age, dicom_to_json, drop_fields_by_length, get_date
from .volume import (
    VOLUME_HANDLERS,
    KeepVolume,
    RandomSlice,
    ReduceVolume,
    SliceAtLocation,
    UniformSample,
    VolumeHandler,
)


try:
    pass
except ImportError:
    print("DICOM operations require pydicom package")
    raise


__version__ = importlib.metadata.version("dicom-utils")


SPAM_WARNING_PATTERNS: Final = {
    ".*Invalid value for VR UI.*",
    ".*It's recommended that you change the Bits Stored value to produce the correct output.*",
}


def filter_spam_warnings(patterns: Iterable[str] = SPAM_WARNING_PATTERNS, action: str = "ignore") -> None:
    r"""Filters certain pydicom warnings that result in spam when manipulating DICOMS in bulk.
    If the environment variable ``SILENCE_PYDICOM_SPAM=1``, this function will be called
    automatically with ``action="ignore"`` when dicom-utils is imported.
    """
    for pattern in patterns:
        warnings.filterwarnings(cast(Literal["ignore"], action), pattern)


SILENCE_ENV_VAR: Final = "SILENCE_PYDICOM_SPAM"
if os.environ.get(SILENCE_ENV_VAR, "") == "1":
    filter_spam_warnings()


__all__ = [
    "__version__",
    "add_patient_age",
    "anonymize",
    "dicom_to_json",
    "DicomFactory",
    "drop_fields_by_length",
    "get_date",
    "filter_spam_warnings",
    "NoImageError",
    "read_dicom_image",
    "VolumeHandler",
    "KeepVolume",
    "SliceAtLocation",
    "UniformSample",
    "ReduceVolume",
    "RandomSlice",
    "VOLUME_HANDLERS",
]

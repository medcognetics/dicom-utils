#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import re
from typing import Optional, Protocol, Union, cast, runtime_checkable

from .helpers import SOPUID, ImageUID, SeriesUID, StudyUID


logger = logging.getLogger(__name__)


@runtime_checkable
class SupportsStudyUID(Protocol):
    StudyInstanceUID: Optional[StudyUID] = None

    def same_study_as(self, other: "SupportsStudyUID") -> bool:
        r"""Checks if this record is part of the same study as record ``other``."""
        return bool(self.StudyInstanceUID) and self.StudyInstanceUID == other.StudyInstanceUID


@runtime_checkable
class SupportsPatientID(Protocol):
    PatientID: Optional[str] = None
    PatientName: Optional[str] = None

    def same_patient_as(self, other: "SupportsPatientID") -> bool:
        r"""Checks if this record references the same patient as record ``other``."""
        if self.PatientID:
            return self.PatientID == other.PatientID
        else:
            return bool(self.PatientName) and self.PatientName == other.PatientName


@runtime_checkable
class SupportsStudyID(Protocol):
    StudyID: Optional[str] = None


@runtime_checkable
class SupportsUID(Protocol):
    SOPInstanceUID: Optional[SOPUID] = None
    SeriesInstanceUID: Optional[SeriesUID] = None

    @property
    def has_uid(self) -> bool:
        r"""Tests if the record has a SeriesInstanceUID or SOPInstanceUID"""
        return bool(self.SeriesInstanceUID or self.SOPInstanceUID)

    def same_uid_as(self, other: "SupportsUID") -> bool:
        r"""Checks if this record is part of the same study as record ``other``."""
        return bool(self.SOPInstanceUID) and bool(other.SOPInstanceUID) and self.SOPInstanceUID == other.SOPInstanceUID

    def get_uid(self, prefer_sop: bool = True) -> ImageUID:
        r"""Gets an image level UID. The UID will be chosen from SeriesInstanceUID and SOPInstanceUID,
        with preference as specified in ``prefer_sop``.
        """
        if not self.has_uid:
            raise AttributeError("DicomFileRecord has no UID")
        if prefer_sop:
            result = self.SOPInstanceUID or self.SeriesInstanceUID
        else:
            result = self.SeriesInstanceUID or self.SOPInstanceUID
        assert result is not None
        return result


@runtime_checkable
class SupportsStudyDate(Protocol):
    StudyDate: Optional[str] = None

    @property
    def StudyYear(self) -> Optional[int]:
        r"""Extracts a year from ``StudyDate``.

        Returns:
            First 4 digits of ``StudyDate`` as an int, or None if a year could not be parsed
        """
        if self.StudyDate and len(self.StudyDate) > 4:
            try:
                return int(self.StudyDate[:4])
            except Exception:
                pass
        return None


@runtime_checkable
class SupportsGenerated(Protocol):
    generated: bool


@runtime_checkable
class SupportsManufacturer(Protocol):
    Manufacturer: Optional[str] = None
    ManufacturerModelName: Optional[str] = None
    ManufacturerModelNumber: Optional[str] = None

    def search_manufacturer_info(self, pattern: Union[str, re.Pattern]) -> Optional[re.Match]:
        pattern = pattern if isinstance(pattern, str) else cast(re.Pattern, re.compile(pattern))
        assert isinstance(pattern, re.Pattern)
        if self.Manufacturer and (m := pattern.search(self.Manufacturer)):
            return m
        elif self.ManufacturerModelName and (m := pattern.search(self.ManufacturerModelName)):
            return m
        elif self.ManufacturerModelNumber and (m := pattern.search(self.ManufacturerModelNumber)):
            return m
        return None
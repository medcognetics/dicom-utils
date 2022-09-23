#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional

import pytest

from dicom_utils.container.protocols import (
    SupportsManufacturer,
    SupportsPatientAge,
    SupportsPatientID,
    SupportsStudyUID,
    SupportsUID,
)


class TestSupportsStudyUID:
    @pytest.fixture
    def factory(self):
        @dataclass
        class Impl(SupportsStudyUID):
            StudyInstanceUID: Optional[str] = None

        return Impl

    @pytest.mark.parametrize(
        "uid1,uid2,exp",
        [
            pytest.param("1.2.3", "1.2.3", True),
            pytest.param(None, "1.2.3", False),
            pytest.param("1.2.3", None, False),
        ],
    )
    def test_same_study_as(self, factory, uid1, uid2, exp):
        s1 = factory(uid1)
        s2 = factory(uid2)
        assert s1.same_study_as(s2) == exp


class TestSupportsPatientID:
    @pytest.fixture
    def factory(self):
        @dataclass
        class Impl(SupportsPatientID):
            PatientID: Optional[str] = None
            PatientName: Optional[str] = None

        return Impl

    @pytest.mark.parametrize(
        "id1,name1,id2,name2,exp",
        [
            pytest.param("P1", None, "P1", None, True),
            pytest.param("P1", None, "P2", None, False),
            pytest.param("P1", None, None, None, False),
            pytest.param(None, None, None, None, False),
            pytest.param("P1", "John Doe", "P2", "John Doe", False),
            pytest.param("P1", "John A. Doe", "P1", "John Alex Doe", True),
        ],
    )
    def test_same_patient_as(self, factory, id1, name1, id2, name2, exp):
        s1 = factory(id1, name1)
        s2 = factory(id2, name2)
        assert s1.same_patient_as(s2) == exp


class TestSupportsUID:
    @pytest.fixture
    def factory(self):
        @dataclass
        class Impl(SupportsUID):
            SOPInstanceUID: Optional[str] = None
            SeriesInstanceUID: Optional[str] = None

        return Impl

    @pytest.mark.parametrize(
        "sop,series,prefer_sop,exp",
        [
            pytest.param("1.2.3", None, True, "1.2.3"),
            pytest.param(None, "1.2.3", True, "1.2.3"),
            pytest.param("1.2.3", "2.3.4", True, "1.2.3"),
            pytest.param("1.2.3", "2.3.4", False, "2.3.4"),
            pytest.param(None, None, True, None, marks=pytest.mark.xfail(raises=AttributeError)),
        ],
    )
    def test_get_uid(self, factory, sop, series, prefer_sop, exp):
        s = factory(sop, series)
        assert s.get_uid(prefer_sop) == exp

    @pytest.mark.parametrize(
        "sop1,series1,sop2,series2,exp",
        [
            pytest.param("1.2.3", None, "1.2.3", None, True),
            pytest.param("1.2.3", None, "2.3.4", None, False),
            pytest.param("1.2.3", "1.2.3", "2.3.4", "1.2.3", False),
            pytest.param(None, "1.2.3", None, "1.2.3", True),
            pytest.param(None, None, None, None, False),
        ],
    )
    def test_same_patient_as(self, factory, sop1, series1, sop2, series2, exp):
        s1 = factory(sop1, series1)
        s2 = factory(sop2, series2)
        assert s1.same_uid_as(s2) == exp


class TestSupportsManufacturer:
    @pytest.fixture
    def factory(self):
        @dataclass
        class Impl(SupportsManufacturer):
            Manufacturer: Optional[str] = None
            ManufacturerModelName: Optional[str] = None
            ManufacturerModelNumber: Optional[str] = None

        return Impl

    @pytest.mark.parametrize(
        "man,model,number,pattern,exp",
        [
            pytest.param(None, None, None, "foo", None),
            pytest.param("man", "model", "number", "m.*", "man"),
            pytest.param(None, "model", "number", "m.*", "model"),
            pytest.param("foo", "bar", "model_num", "m.*", "model_num"),
        ],
    )
    def test_search_manufacturer_info(self, factory, man, model, number, pattern, exp):
        s = factory(man, model, number)
        m = s.search_manufacturer_info(pattern)
        if exp is None:
            assert m is None
        else:
            assert m.group() == exp


class TestSupportsPatientAge:
    @pytest.fixture
    def factory(self):
        @dataclass
        class Impl(SupportsPatientAge):
            PatientAge: Optional[str] = None
            PatientBirthDate: Optional[str] = None

        return Impl

    @pytest.mark.parametrize(
        "age,exp",
        [
            pytest.param("095Y", 95),
            pytest.param("020Y", 20),
            pytest.param("99Y", 99),
            pytest.param("32", 32),
            pytest.param("", None),
            pytest.param(None, None),
        ],
    )
    def test_patient_age(self, factory, age, exp):
        s = factory(age)
        assert s.patient_age == exp

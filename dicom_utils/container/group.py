#!/usr/bin/env python
# -*- coding: utf-8 -*-


from functools import partial
from pathlib import Path
from typing import Callable, Hashable, Optional, ParamSpec, TypeVar

from registry import Registry

# Type checking fails when dataclass attr name matches a type alias.
# Import types under a different alias
from .helpers import StudyUID
from .record import FileRecord


P = ParamSpec("P")
H = TypeVar("H", bound=Hashable)


GroupFunction = Callable[[FileRecord], Hashable]
GROUP_REGISTRY = Registry("group", bound=Callable[..., Hashable])


@GROUP_REGISTRY(name="study-uid")
def group_by_study_instance_uid(rec: FileRecord) -> Optional[StudyUID]:
    return getattr(rec, "StudyInstanceUID", None)


@GROUP_REGISTRY(name="parent")
def group_by_parent(rec: FileRecord, level: int = 0) -> Path:
    return rec.path.parents[level]


for i in range(3):
    GROUP_REGISTRY(partial(group_by_parent, level=i + 1), name=f"parent-{i+1}")


@GROUP_REGISTRY(name="patient-id")
def group_by_patient_id(rec: FileRecord) -> Optional[str]:
    return getattr(rec, "PatientID", None)


@GROUP_REGISTRY(name="study-date")
def group_by_study_date(rec: FileRecord) -> Optional[str]:
    return getattr(rec, "StudyDate", None)

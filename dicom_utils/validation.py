#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from dataclasses import dataclass, field
from enum import IntEnum
from functools import cached_property, partial
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Final, Generic, Iterable, List, Optional, Sequence, Sized, Tuple, TypeVar, Union

import numpy as np
import pydicom

from .dicom import Dicom, TransferSyntaxUIDs
from .metadata import MAX_FIELD_LENGTH
from .tags import Tag
from .types import PhotometricInterpretation


try:
    from colorama import Fore  # type: ignore
except Exception:
    Fore = None

T = TypeVar("T")


COLUMNS: Final = ("Tag", "Priority", "State", "Message")


def color_str(x: Any, color: str) -> str:
    if Fore is None:
        return str(x)
    elif not hasattr(Fore, color):
        raise AttributeError(f"Color {color} is not a valid Fore color")
    else:
        c = getattr(Fore, color)
        return f"{c}{x}{Fore.RESET}"


ValidationFunction = Callable[[T], Optional[str]]


def min_max_validator(min: float = float("-inf"), max: float = float("inf")) -> ValidationFunction:
    def func(val: Any) -> Optional[str]:
        try:
            if min <= val <= max:
                return None
            return "Value {val} outside interval [{min}, {max}]"
        except Exception:
            return f"Unhandled value {val}"

    return func


def int_validator(val: Any) -> Optional[str]:
    try:
        if int(val) == float(val):
            return None
    except Exception:
        pass
    return f"Value {val} is not an int"


def float_validator(val: Any) -> Optional[str]:
    try:
        float(val)
        return None
    except Exception:
        return f"Value {val} is not a float"


def age_validator(val: Any) -> Optional[str]:
    try:
        parsed_age = int(re.sub("[^0-9]", "", val))
    except Exception:
        return f"Unparsable age {val}"
    range_result = min_max_validator(20, 120)(parsed_age)
    return range_result


def transfer_syntax_validator(val: Any) -> Optional[str]:
    if val not in TransferSyntaxUIDs.keys():
        return f"Unknown TransferSyntaxUID {val}"
    elif "lossy" in (fullname := TransferSyntaxUIDs[val].lower()):
        return f"Detected lossy compression {fullname}"
    return None


def pixel_data_validator(val: Any) -> Optional[str]:
    if not len(val):
        return "Pixel data was empty"
    return None


def photometric_validator(val: Any) -> Optional[str]:
    pm = PhotometricInterpretation.from_str(val)
    if not pm:
        return f"Invalid value {val}"
    return None


def bits_validator(val: Any) -> Optional[str]:
    int_val = int_validator(val)
    if int_val is not None:
        return int_val
    val = int(val)
    if val <= 8:
        return f"Bit value {val} suggests low dynamic range"
    return None


class ValidationLevel(IntEnum):
    IGNORE = 0
    PRESENT = 1
    VALID = 2
    ABSENT = 3


class Priority(IntEnum):
    IGNORE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAl = 4

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    @classmethod
    def all_values(cls) -> Sequence["Priority"]:
        return list(cls)


@dataclass
class ValidationResult(Generic[T]):
    tag: Tag
    name: str
    level: ValidationLevel
    priority: Priority
    is_valid: bool
    value: Optional[T]
    msg: str

    @property
    def is_present(self) -> bool:
        if isinstance(self.value, (str, Sized)):
            return bool(len(self.value))
        return self.value is not None

    def __add__(self, other: "ValidationResult") -> "ValidationResult":
        return self if self.is_valid else other

    @property
    def result_level(self) -> ValidationLevel:
        if self.is_valid:
            return ValidationLevel.VALID
        elif self.is_present:
            return ValidationLevel.PRESENT
        else:
            return ValidationLevel.ABSENT

    @property
    def is_passing(self) -> bool:
        return self.result_level is not ValidationLevel.ABSENT and self.result_level >= self.level

    @property
    def _color(self) -> str:
        if self.is_valid:
            result_level = ValidationLevel.VALID
        elif self.is_present:
            result_level = ValidationLevel.PRESENT
        else:
            result_level = ValidationLevel.IGNORE

        if result_level >= self.level:
            return "GREEN"
        elif self.level == ValidationLevel.VALID:
            return "YELLOW"
        else:
            return "RED"

    @property
    def _passing_str(self) -> str:
        if self.is_passing:
            return "PASS"
        elif not self.is_present:
            return "MISSING"
        elif self.level == ValidationLevel.ABSENT:
            return "FAIL"
        else:
            return "INVALID"

    def to_line(self, fmt: str, color: bool = False) -> str:
        str_fn = partial(color_str, color=self._color) if color else str
        result = fmt.format(
            str_fn(str(self.tag)),
            str_fn(self.priority),
            str_fn(self._passing_str),
            str_fn(self.msg),
        )
        return result

    def header(self, fmt: str, color: bool = False) -> str:
        str_fn = partial(color_str, color="WHITE") if color else str
        result = fmt.format(*(str_fn(x) for x in COLUMNS))
        return result

    def column_widths(self, color: bool = False) -> List[int]:
        str_fn = partial(color_str, color=self._color) if color else str
        values = [
            str_fn(str(self.tag)),
            str_fn(self.priority),
            str_fn(self._passing_str),
            str_fn(self.msg),
        ]
        result = [max(len(v), len(color_str(c, "WHITE"))) for v, c in zip(values, COLUMNS)]
        return result


@dataclass
class ValidationTarget(Generic[T]):
    tag: Tag
    level: ValidationLevel
    priority: Priority = Priority.MEDIUM
    validator: ValidationFunction = field(default=lambda _: None, repr=False, hash=False)

    def __lt__(self, other: Union["ValidationTarget", Iterable["ValidationTarget"]]):
        if isinstance(other, ValidationTarget):
            return self.priority < other.priority
        elif isinstance(other, Sequence):
            return self.priority < max(x.priority for x in other)
        raise TypeError(other)

    def __gt__(self, other: Union["ValidationTarget", Iterable["ValidationTarget"]]):
        if isinstance(other, ValidationTarget):
            return self.priority > other.priority
        elif isinstance(other, Sequence):
            return self.priority > max(x.priority for x in other)
        raise TypeError(other)

    def __eq__(self, other: Union["ValidationTarget", Iterable["ValidationTarget"]]):
        if isinstance(other, ValidationTarget):
            return self.priority == other.priority
        elif isinstance(other, Sequence):
            return self.priority == max(x.priority for x in other)
        return False

    @cached_property
    def name(self) -> str:
        try:
            return pydicom.DataElement(self.tag, "US", None).name
        except Exception:
            return "Unnamed"

    def get_value(self, dcm: Dicom) -> Optional[T]:
        val = dcm.get(self.tag, None) or dcm.file_meta.get(self.tag, None)
        return val.value if val is not None else None

    def is_present(self, dcm: Dicom) -> bool:
        val = self.get_value(dcm)
        if isinstance(val, (str, Sized)):
            return bool(len(val))
        return val is not None

    def is_valid(self, dcm: Dicom) -> bool:
        result = self.is_present(dcm) and self.validator(self.get_value(dcm)) is None
        return result

    def validation_msg(self, dcm: Dicom) -> str:
        if not self.is_present(dcm):
            return "Field not present"
        result = self.validator(self.get_value(dcm))
        if result is not None:
            return result
        result = str(self.get_value(dcm))
        if len(result) <= MAX_FIELD_LENGTH:
            return result
        return "OK"

    def __call__(self, dcm: Dicom) -> ValidationResult:
        valid = self.is_valid(dcm)
        valid_msg = self.validation_msg(dcm)
        value = self.get_value(dcm)

        result = ValidationResult(
            self.tag,
            self.name,
            self.level,
            self.priority,
            is_valid=valid,
            value=value,
            msg=valid_msg,
        )
        return result


DEFAULT_TARGETS = [
    ValidationTarget(Tag.PatientIdentityRemoved, ValidationLevel.VALID, Priority.HIGH),
    ValidationTarget(Tag.StudyInstanceUID, ValidationLevel.VALID, Priority.HIGH),
    ValidationTarget(Tag.SeriesInstanceUID, ValidationLevel.VALID, Priority.HIGH),
    ValidationTarget(Tag.SOPInstanceUID, ValidationLevel.VALID, Priority.HIGH),
    ValidationTarget(Tag.TransferSyntaxUID, ValidationLevel.VALID, Priority.HIGH, validator=transfer_syntax_validator),
    ValidationTarget(Tag.PatientAge, ValidationLevel.VALID, validator=age_validator),
    ValidationTarget(Tag.Manufacturer, ValidationLevel.VALID),
    ValidationTarget(Tag.ManufacturerModelName, ValidationLevel.VALID),
    ValidationTarget(Tag.DeviceSerialNumber, ValidationLevel.VALID),
    ValidationTarget(Tag.Rows, ValidationLevel.VALID, validator=min_max_validator(min=1)),
    ValidationTarget(Tag.Columns, ValidationLevel.VALID, validator=min_max_validator(min=1)),
    ValidationTarget(Tag.PixelData, ValidationLevel.VALID, Priority.HIGH, validator=pixel_data_validator),
    (
        ValidationTarget(Tag.Laterality, ValidationLevel.VALID, Priority.HIGH),
        ValidationTarget(Tag.ImageLaterality, ValidationLevel.VALID, Priority.HIGH),
    ),
    ValidationTarget(Tag.ViewPosition, ValidationLevel.VALID, Priority.HIGH),
    ValidationTarget(Tag.PatientSex, ValidationLevel.VALID),
    ValidationTarget(Tag.EthnicGroup, ValidationLevel.VALID),
    ValidationTarget(Tag.BreastImplantPresent, ValidationLevel.VALID, Priority.HIGH),
    ValidationTarget(Tag.PhotometricInterpretation, ValidationLevel.VALID, Priority.HIGH, photometric_validator),
    ValidationTarget(Tag.PixelSpacing, ValidationLevel.VALID, Priority.LOW),
    ValidationTarget(Tag.BitsStored, ValidationLevel.VALID, Priority.LOW, bits_validator),
    ValidationTarget(Tag.PartialView, ValidationLevel.PRESENT, Priority.LOW),
    ValidationTarget(Tag.LossyImageCompression, ValidationLevel.VALID, Priority.LOW),
    ValidationTarget(Tag.Density, ValidationLevel.VALID, Priority.LOW),
    ValidationTarget(Tag.WindowCenter, ValidationLevel.IGNORE, Priority.LOW),
    ValidationTarget(Tag.WindowWidth, ValidationLevel.IGNORE, Priority.LOW),
]


class Validator:
    def __init__(self, targets: Sequence[Union[ValidationTarget, Tuple[ValidationTarget, ...]]]):
        self.targets = list(sorted(targets, reverse=True))

    def __add__(self, other: "Validator") -> "Validator":
        if not isinstance(other, Validator):
            raise TypeError(f"Expected Validator for other, found {type(other)}")
        return Validator(list(chain(self.targets, other.targets)))

    def validate_dicom_object(self, dcm: Dicom) -> Sequence[ValidationResult]:
        result: List[ValidationResult] = []
        for target in self.targets:
            if isinstance(target, Iterable):
                all_tag_results = [t(dcm) for t in target]
                tag_result = sum(all_tag_results, all_tag_results[0])
            else:
                tag_result = target(dcm)
            result.append(tag_result)
        return result

    def validate_dicom_file(self, path: Path) -> Sequence[ValidationResult]:
        if not path.is_file():
            raise FileNotFoundError(path)
        with pydicom.dcmread(path) as dcm:
            result = self.validate_dicom_object(dcm)
        return result

    @classmethod
    def default_validator(cls) -> "Validator":
        return cls(DEFAULT_TARGETS)

    @staticmethod
    def _format(
        results: Sequence[ValidationResult],
        delim: str = "\t",
        color: bool = False,
        padding: int = 2,
    ) -> str:
        lens = np.array([x.column_widths(color) for x in results])
        lens = lens.max(axis=0)
        lens += padding
        return delim.join([f"{{:<{x}}}" for x in lens])

    @classmethod
    def report_string(
        cls, results: Sequence[ValidationResult], failing_only: bool = False, color: bool = False
    ) -> None:
        passing = 0
        non_passing = {k: 0 for k in Priority.all_values()}
        fmt = cls._format(results, color=color)
        print(results[0].header(fmt, color=color))
        for result in results:
            if result.is_passing and failing_only:
                pass
            else:
                print(result.to_line(fmt, color=color))

            if result.is_passing:
                passing += 1
            else:
                non_passing[result.priority] += 1
        print("\n=== Summary === ")
        print(f"Passing: {passing}")
        print(f"Failing: {sum(v for v in non_passing.values())}")
        for k, v in non_passing.items():
            if v == 0:
                continue
            print(f"  {k.name}: {v}")

    @classmethod
    def all_passing(cls, results: Sequence[ValidationResult]) -> bool:
        return all(r.is_passing for r in results)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import typing
from copy import copy, deepcopy
from functools import partial
from inspect import signature
from itertools import product
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, TypeVar, Union, cast

import numpy as np
import pydicom
from numpy.random import default_rng
from pydicom import DataElement, Dataset, FileDataset, Sequence
from pydicom.data import get_testdata_file
from pydicom.valuerep import VR

from .tags import Tag


T = TypeVar("T", bound="DicomFactory")
U = TypeVar("U")


def args_from_dicom(func: Callable[..., U], dicom: FileDataset) -> Callable[..., U]:
    sig = signature(func)
    kwargs: Dict[str, Any] = {}
    for name in sig.parameters.keys():
        if not hasattr(Tag, name) or not hasattr(dicom, name):
            continue
        tag = getattr(Tag, name)
        kwargs[name] = dicom[tag].value
    return partial(func, **kwargs)


# NOTE: This class should not use any dicom-utils methods in its implementation, as it will
# be used in the testing of dicom-utils
class DicomFactory:
    r"""Factory class for creating DICOM objects for unit tests.

    Args:
        proto:
            A prototype DICOM on which defaults will be based. Can be a DICOM FileDataset object,
            a path to a DICOM file, or a string with a pydicom testdata file.

        seed:
            Seed for random number generation

    Keyword Args:
        Tag value overrides
    """
    dicom: FileDataset

    def __init__(
        self,
        proto: Union[PathLike, FileDataset, str] = "CT_small.dcm",
        seed: int = 42,
        allow_nonproto_tags: bool = True,
        **kwargs,
    ):
        self.seed = int(seed)
        self.rng = default_rng(self.seed)
        self.allow_nonproto_tags = allow_nonproto_tags
        self.overrides = kwargs
        if isinstance(proto, (PathLike, str)):
            self.path = Path(proto)
            if not self.path.is_file():
                self.path = Path(cast(str, get_testdata_file(str(proto))))
            self.dicom = pydicom.dcmread(self.path)

        elif isinstance(proto, FileDataset):
            self.path = None
            self.dicom = proto

        else:
            raise TypeError(f"`proto` should be PathLike or FileDataset, found {type(proto)}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.path})"

    def __enter__(self: T) -> T:
        result = copy(self)
        result.seed = self.seed
        result.rng = default_rng(self.seed)
        return result

    def __call__(self, seed: Optional[int] = None, **kwargs) -> FileDataset:
        self.rng if seed is None else default_rng(seed)
        dcm = deepcopy(self.dicom)

        overrides = {**self.overrides, **kwargs}
        for tag_name, value in overrides.items():
            tag = getattr(Tag, tag_name)
            if tag in dcm.keys():
                dcm[tag].value = value
            elif tag in dcm.file_meta:
                dcm.file_meta[tag].value = value
            else:
                elem = self.data_element(tag, value)
                dcm[tag] = elem

        if not self.allow_nonproto_tags:
            for tag_name in overrides.keys():
                if tag_name not in self.dicom:
                    del dcm[tag_name]

        return dcm

    @classmethod
    def data_element(cls, tag: Tag, value: Any, vr: Optional[Union[VR, str]] = None, **kwargs) -> DataElement:
        if vr is None:
            vr = cls.suggest_vr(tag, value)
        else:
            vr = VR(vr)
        return DataElement(tag, vr, value, **kwargs)

    @classmethod
    def pixel_array(
        cls,
        Rows: int,
        Columns: int,
        NumberOfFrames: int = 1,
        BitsStored: int = 16,
        BitsAllocated: int = 14,
        PhotometricInterpretation: str = "MONOCHROME2",
        seed: int = 42,
    ) -> np.ndarray:
        low = 0
        high = BitsAllocated
        channels = 1 if PhotometricInterpretation.startswith("MONOCHROME") else 3
        size = tuple(x for x in (channels, NumberOfFrames, Rows, Columns) if x > 1)
        rng = default_rng(seed)
        return rng.integers(low, high, size, dtype=np.uint16)

    @classmethod
    def pixel_array_from_dicom(
        cls,
        dcm: FileDataset,
        seed: int = 42,
    ) -> np.ndarray:
        func = args_from_dicom(cls.pixel_array, dcm)
        return func(seed=seed)

    @classmethod
    def code_sequence(cls, *meanings: str) -> Sequence:
        codes: List[Dataset] = []
        for meaning in meanings:
            vc = Dataset()
            vc[Tag.CodeMeaning] = cls.data_element(Tag.CodeMeaning, meaning, "ST")
            codes.append(vc)
        return Sequence(codes)

    @classmethod
    def suggest_vr(cls, tag: Tag, value: Any) -> VR:
        name = tag.name
        if isinstance(value, typing.Sequence) and not isinstance(value, str):
            return VR("SQ")
        elif isinstance(value, int):
            return VR("UL")
        elif name.endswith("UID"):
            return VR("UI")
        elif name.endswith("Age"):
            return VR("AS")
        elif name.endswith("Date"):
            return VR("DA")
        elif name.endswith("Time"):
            return VR("TM")
        return VR("ST")

    @classmethod
    def ffdm_factory(cls: Type[T], seed: int = 42, **kwargs) -> T:
        overrides = {
            "Modality": "MG",
            "ImageLaterality": "L",
            "ViewPosition": "CC",
        }
        overrides.update(kwargs)
        return cls(seed=seed, **overrides)

    @classmethod
    def tomo_factory(cls: Type[T], seed: int = 42, **kwargs) -> T:
        overrides = {
            "NumberOfFrames": 3,
        }
        overrides.update(kwargs)
        return cls.ffdm_factory(seed=seed, **overrides)

    @classmethod
    def synth_factory(cls: Type[T], seed: int = 42, **kwargs) -> T:
        overrides = {
            "SeriesDescription": "S-view",
        }
        overrides.update(kwargs)
        return cls.ffdm_factory(seed=seed, **overrides)

    @classmethod
    def ultrasound_factory(cls: Type[T], seed: int = 42, **kwargs) -> T:
        overrides = {
            "Modality": "US",
        }
        overrides.update(kwargs)
        return cls(seed=seed, **overrides)

    @classmethod
    def complete_mammography_case_factory(
        cls,
        seed: int = 42,
        implants: bool = False,
        spot_compression: bool = False,
        types: Iterable[str] = ("ffdm", "synth", "tomo"),
        lateralities: Iterable[str] = ("L", "R"),
        views: Iterable[str] = ("MLO", "CC"),
        dates: Iterable[str] = ("01012000",),
        **kwargs,
    ) -> "ConcatFactory":
        IMPLANTS = (False, True) if implants else (False,)
        SPOT = (False, True) if spot_compression else (False,)

        lateralities = set(lateralities)
        types = set(types)
        views = set(views)

        factories: List[DicomFactory] = []
        iterator = product(lateralities, views, types, IMPLANTS, SPOT, dates)
        for i, (laterality, view, mtype, implant, spot, date) in enumerate(iterator):
            meanings: List[str] = []
            if not implant and implants:
                meanings.append("implant displaced")
            if spot:
                meanings.append("spot compression")
            codes = cls.code_sequence(*meanings)
            overrides = {
                **kwargs,
                "ImageLaterality": laterality,
                "ViewPosition": view,
                "BreastImplantPresent": "YES" if implant else "NO",
                "ViewModifierCodeSequence": codes,
                "SOPInstanceUID": f"sop-{i}",
                "SeriesInstanceUID": f"series-{i}",
                "StudyDate": date,
            }
            if not hasattr(cls, (factory_name := f"{mtype}_factory")):
                raise ValueError(f"Factory function {factory_name} not found")
            factory = getattr(cls, factory_name)(seed, **overrides)
            factories.append(factory)

        if "ultraound" in types:
            factory = cls.ultrasound_factory(seed, SOPInstanceUID="sop-us", SeriesInstanceUID="series-us", **kwargs)
            factories.append(factory)
        return ConcatFactory(*factories)

    @classmethod
    def save_dicoms(cls, path: PathLike, dicoms: Iterable[FileDataset]) -> List[Path]:
        root = Path(path)
        results: List[Path] = []
        for i, dcm in enumerate(dicoms):
            path = Path(root, f"D{i}.dcm")
            dcm.save_as(path)
            results.append(path)
        return results


class ConcatFactory:
    def __init__(self, *factories: DicomFactory):
        self.factories = list(factories)

    def __call__(self, seed: Optional[int] = None, **kwargs) -> List[FileDataset]:
        return [factory(seed, **kwargs) for factory in self.factories]

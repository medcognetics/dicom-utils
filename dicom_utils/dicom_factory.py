#!/usr/bin/env python
# -*- coding: utf-8 -*-

import typing
from abc import ABC, abstractmethod
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
from registry import Registry

from .dicom import set_pixels
from .tags import Tag


T = TypeVar("T", bound="DicomFactory")
C = TypeVar("C", bound="ConcatFactory")
U = TypeVar("U")
Proto = Union[PathLike, FileDataset, str]
DEFAULT_PROTO = "CT_small.dcm"
FACTORY_REGISTRY = Registry("factories")


def args_from_dicom(func: Callable[..., U], dicom: FileDataset) -> Callable[..., U]:
    sig = signature(func)
    kwargs: Dict[str, Any] = {}
    for name in sig.parameters.keys():
        if not hasattr(Tag, name) or not hasattr(dicom, name):
            continue
        tag = getattr(Tag, name)
        kwargs[name] = dicom[tag].value
    return partial(func, **kwargs)


class BaseFactory(ABC):
    @abstractmethod
    def __call__(self, seed: Optional[int] = None, **kwargs) -> FileDataset:
        raise NotImplementedError

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
        high = 2**BitsAllocated - 1
        channels = 1 if PhotometricInterpretation.startswith("MONOCHROME") else 3
        size = tuple(x for x in (channels, NumberOfFrames, Rows, Columns) if x > 1)
        rng = default_rng(seed)
        return rng.integers(low, high, size, dtype=np.uint16)

    @classmethod
    def random_uid(cls, length: int = 6, seed: int = 42) -> str:
        rng = default_rng(seed)
        return "".join(str(x) for x in rng.integers(0, 10, length))

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
        if name.endswith("PixelSpacing"):
            return VR("DS")
        elif isinstance(value, typing.Sequence) and not isinstance(value, str):
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
    def save_dicoms(cls, path: PathLike, dicoms: Iterable[FileDataset]) -> List[Path]:
        root = Path(path)
        results: List[Path] = []
        for i, dcm in enumerate(dicoms):
            path = Path(root, f"D{i}.dcm")
            dcm.save_as(path)
            results.append(path)
        return results


# NOTE: This class should not use any dicom-utils methods in its implementation, as it will
# be used in the testing of dicom-utils
@FACTORY_REGISTRY(name="dicom")
class DicomFactory(BaseFactory):
    r"""Factory class for creating DICOM objects for unit tests.

    Args:
        proto:
            A prototype DICOM on which defaults will be based. Can be a DICOM FileDataset object,
            a path to a DICOM file, or a string with a pydicom testdata file.

        seed:
            Seed for random number generation

        pixels:
            Whether to include pixel data in the generated DICOM. If False, ``dcm.PixelData`` will be set to ``None``.

    Keyword Args:
        Tag value overrides
    """

    dicom: FileDataset

    def __init__(
        self,
        proto: Proto = DEFAULT_PROTO,
        seed: int = 42,
        allow_nonproto_tags: bool = True,
        pixels: bool = True,
        **kwargs,
    ):
        self.seed = int(seed)
        self.rng = default_rng(self.seed)
        self.allow_nonproto_tags = allow_nonproto_tags
        self.pixels = pixels
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

    def __add__(self: T, other: T) -> T: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.path})"

    def __enter__(self: T) -> T:
        result = copy(self)
        result.seed = self.seed
        result.rng = default_rng(self.seed)
        return result

    def __call__(self, seed: Optional[int] = None, pixels: Optional[bool] = None, **kwargs) -> FileDataset:
        self.rng if seed is None else default_rng(seed)
        dcm = deepcopy(self.dicom)

        # set overrides
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

        # Rows/Columns may have been changed without changing PixelData. This will cause
        # an error when reading `dcm.pixel_array`. Try reading pixel_array and if it fails
        # update pixels with new data of the correct shape. We will only perform the pixel update
        # if requested because it can be slow.
        pixels = self.pixels if pixels is None else pixels
        if pixels:
            try:
                dcm.pixel_array
            except ValueError:
                arr = self.pixel_array(
                    dcm.Rows,
                    dcm.Columns,
                    dcm.NumberOfFrames,
                    dcm.BitsStored,
                    dcm.BitsAllocated,
                    dcm.PhotometricInterpretation,
                    seed=self.seed,
                )
                dcm = set_pixels(dcm, arr, dcm.file_meta.TransferSyntaxUID)
        elif hasattr(dcm, "PixelData"):
            del dcm.PixelData

        # if requested, delete any tags not in the proto
        if not self.allow_nonproto_tags:
            for tag_name in overrides.keys():
                if tag_name not in self.dicom:
                    del dcm[tag_name]

        return dcm


FFDMFactory = partial(DicomFactory, Modality="MG", Laterality="L", ViewPosition="CC", NumberOfFrames=1)
TOMOFactory = partial(FFDMFactory, NumberOfFrames=3)
SynthFactory = partial(FFDMFactory, SeriesDescription="S-view")
UltrasoundFactory = partial(DicomFactory, Modality="US")
FACTORY_REGISTRY(FFDMFactory, name="ffdm")
FACTORY_REGISTRY(TOMOFactory, name="tomo")
FACTORY_REGISTRY(SynthFactory, name="synth")
FACTORY_REGISTRY(UltrasoundFactory, name="ultrasound")


class ConcatFactory(BaseFactory):
    def __init__(self, factories: Iterable[Union[str, DicomFactory, "ConcatFactory"]]):
        self.factories = [
            f if isinstance(f, (DicomFactory, ConcatFactory)) else FACTORY_REGISTRY.get(f)() for f in factories
        ]

    def __add__(self: C, other: C) -> C:
        result = copy(self)
        result.factories = self.factories + other.factories
        return result

    def __call__(self, seed: Optional[int] = None, pixels: Optional[bool] = None, **kwargs) -> List[FileDataset]:
        result: List[FileDataset] = []
        for factory in self.factories:
            _result = factory(seed=seed, pixels=pixels, **kwargs)
            if isinstance(_result, FileDataset):
                result.append(_result)
            else:
                result = result + _result
        return result


@FACTORY_REGISTRY(name="mammo-case")
class CompleteMammographyStudyFactory(ConcatFactory):
    def __init__(
        self,
        proto: Proto = DEFAULT_PROTO,
        seed: int = 42,
        allow_nonproto_tags: bool = True,
        implants: bool = False,
        spot_compression: bool = False,
        types: Iterable[str] = ("ffdm", "synth", "tomo"),
        lateralities: Iterable[str] = ("L", "R"),
        views: Iterable[str] = ("MLO", "CC"),
        **kwargs,
    ):
        IMPLANTS = (False, True) if implants else (False,)
        SPOT = (False, True) if spot_compression else (False,)

        lateralities = set(lateralities)
        types = set(types)
        views = set(views)

        factories: List[DicomFactory] = []
        iterator = product(lateralities, views, types, IMPLANTS, SPOT)
        for i, (laterality, view, mtype, implant, spot) in enumerate(iterator):
            meanings: List[str] = []
            if not implant and implants:
                meanings.append("implant displaced")
            if spot:
                meanings.append("spot compression")
            codes = DicomFactory.code_sequence(*meanings)
            overrides = {
                **kwargs,
                "ImageLaterality": laterality,
                "ViewPosition": view,
                "BreastImplantPresent": "YES" if implant else "NO",
                "ViewModifierCodeSequence": codes,
                "SOPInstanceUID": f"sop-{self.random_uid(seed=seed+i)}-{i}",
                "SeriesInstanceUID": f"series-{self.random_uid(seed=seed+i)}-{i}",
            }
            factory = cast(Type[DicomFactory], FACTORY_REGISTRY.get(mtype))(
                proto=proto,
                seed=seed,
                allow_nonproto_tags=allow_nonproto_tags,
                **overrides,
            )
            factories.append(factory)
        super().__init__(factories)

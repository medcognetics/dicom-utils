from dataclasses import dataclass
from enum import Enum
from typing import Final, Iterator, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np
from numpy import ndarray

from .dicom import read_dicom_image
from .logging import logger
from .types import Dicom, DicomAttributeSequence


Bbox = Tuple[int, int, int, int]
presentation_modality: Final[str] = "PR"


class Form(Enum):
    CIRCLE = "CIRCLE"
    ELLIPSE = "ELLIPSE"
    POLYLINE = "POLYLINE"


@dataclass
class Annotation:
    """Store an annotation with corresponding DICOM filename"""

    def __init__(self, sop_uids: List[str], data: List[float], form: Form):
        self.uids = sop_uids
        self.form = Form(form)
        self.trace = dicom_trace_to_bbox(data, self.form)
        self.is_rectangle = True  # TODO Add non-rectangular trace support

    def __repr__(self):
        return (
            f"<SOPInstanceUIDs: {self.uids}, form: {self.form}, trace: {self.trace}, is rectangle: {self.is_rectangle}>"
        )


@dataclass
class DicomImage:
    """Store DICOM image pixels with associated metadata"""

    pixels: ndarray
    uid: str

    @classmethod
    def from_dicom(cls, dicom: Dicom) -> "DicomImage":
        pixels = to_rgb(read_dicom_image(dicom))
        return cls(pixels, dicom.SOPInstanceUID)

    @property
    def is_single_slice(self) -> bool:
        return self.pixels.shape[0] == 1

    def __repr__(self):
        return f"<SOPInstanceUID: {self.uid}, shape: {self.pixels.shape}>"


class GraphicItem(NamedTuple):
    data: List[float]
    form: Form

    def __add__(self, other: "GraphicItem") -> "GraphicItem":
        assert self.form == other.form == Form.POLYLINE, "Addition is only defined for POLYLINE items"
        return GraphicItem(self.data + other.data, self.form)


def to_collage(images: List[ndarray]) -> ndarray:
    num_images = len(images)

    assert num_images != 0, "There must be at least one image."
    assert all(len(i.shape) == 4 for i in images), "The images must have 4 dimensions."

    image_chns, _, max_image_rows, max_image_cols = np.array([i.shape for i in images]).max(axis=0)

    collage_rows = 1 if num_images < 3 else 2
    collage_cols = int(num_images / collage_rows + 0.5)

    assert all(i.shape[1] == 3 for i in images)
    collage = np.zeros((image_chns, 3, collage_rows * max_image_rows, collage_cols * max_image_cols))

    for i, image in enumerate(images):
        row = int(i >= collage_cols)
        col = i % collage_cols
        start_row = row * max_image_rows
        start_col = col * max_image_cols
        image_chns, _, image_rows, image_cols = image.shape
        collage[:image_chns, :, start_row : start_row + image_rows, start_col : start_col + image_cols] = image

    return collage


def rint(x: float) -> int:
    return int(round(x))


def dicom_ellipse_to_bbox(data: List[float]) -> Bbox:
    assert len(data) != 4 * 3, "This function is not implemented to support 3D (x,y,z) coordinates"
    assert len(data) == 4 * 2, f"Invalid number of data points ({len(data)}) for a DICOM ellipse"
    major_x0, major_y0, major_x1, major_y1, minor_x0, minor_y0, minor_x1, minor_y1 = data
    xs = [rint(v) for v in [major_x0, major_x1, minor_x0, minor_x1]]
    ys = [rint(v) for v in [major_y0, major_y1, minor_y0, minor_y1]]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return x0, y0, x1, y1


def dicom_circle_to_bbox(data: List[float]) -> Bbox:
    assert len(data) != 2 * 3, "This function is not implemented to support 3D (x,y,z) coordinates"
    assert len(data) == 2 * 2, f"Invalid number of data points ({len(data)}) for a DICOM circle"
    x_center, y_center, x_border, y_border = data
    radius = np.sqrt((x_center - x_border) ** 2 + (y_center - y_border) ** 2)
    x0, y0, x1, y1 = [rint(v) for v in [x_center - radius, y_center - radius, x_center + radius, y_center + radius]]
    return x0, y0, x1, y1


def dicom_polylines_to_bbox(data: List[float]) -> Bbox:
    assert len(data) % 2 == 0, "This function is not implemented to support 3D (x,y,z) coordinates"
    xs = [rint(x) for x in data[::2]]
    ys = [rint(x) for x in data[1::2]]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return x0, y0, x1, y1


def dicom_trace_to_bbox(data: List[float], form: Form) -> Bbox:
    if form == Form.CIRCLE:
        return dicom_circle_to_bbox(data)
    elif form == Form.ELLIPSE:
        return dicom_ellipse_to_bbox(data)
    elif form == Form.POLYLINE:
        return dicom_polylines_to_bbox(data)
    else:
        raise Exception(f"Parsing is not supported for {form}")


def chw_to_hwc(image: ndarray) -> ndarray:
    """CxHxW -> HxWxC"""
    return np.rollaxis(image, 0, 3).copy()


def hwc_to_chw(image: ndarray) -> ndarray:
    """HxWxC -> CxHxW"""
    return np.rollaxis(image, 2, 0).copy()


def draw_bbox(image: ndarray, bbox: Bbox, color: Tuple[float, float, float] = (0, 1, 0), thickness: int = 30) -> None:
    x0, y0, x1, y1 = bbox

    for i, slice in enumerate(image):
        slice = chw_to_hwc(slice)
        cv2.rectangle(slice, (x0, y0), (x1, y1), color, thickness)
        image[i] = hwc_to_chw(slice)


def dcms_to_images(dcms: List[Dicom]) -> Iterator[DicomImage]:
    for dcm in dcms:
        try:
            yield DicomImage.from_dicom(dcm)
        except Exception as e:
            logger.info(e)


def distance(a: List[float], b: List[float]) -> float:
    assert len(a) == len(b), "Expected lists of equal length for calculating distance"
    return sum((u - v) ** 2 for u, v in zip(a, b))


def polylines_are_contiguous(a: List[float], b: List[float]) -> bool:
    a_start = a[:2]
    a_stop = a[-2:]
    b_start = b[:2]
    return distance(a_start, a_stop) > distance(a_stop, b_start)


def are_matching_polylines(a: GraphicItem, b: GraphicItem) -> bool:
    return a.form == b.form == Form.POLYLINE and polylines_are_contiguous(a.data, b.data)


def group_polylines(graphic_items: List[GraphicItem]) -> Iterator[GraphicItem]:
    """Consecutive polyline traces may have been recorded separately when they were intended to be part of one single
    trace. Identify this situation and combine polylines accordingly."""
    while graphic_items:
        item = graphic_items.pop(0)

        while graphic_items and are_matching_polylines(item, graphic_items[0]):
            item = item + graphic_items.pop(0)

        yield item


def gen_graphic_items(graphic_objects: DicomAttributeSequence) -> Iterator[GraphicItem]:
    for graphic_object in graphic_objects:
        assert graphic_object.GraphicAnnotationUnits == "PIXEL"
        yield GraphicItem(data=graphic_object.GraphicData, form=Form(graphic_object.GraphicType))


def dcm_to_annotations(dcm: Dicom, target_sop_uid: Optional[str] = None) -> Iterator[Annotation]:
    for graphic_annotation in dcm.get("GraphicAnnotationSequence", []):
        referenced_uids = [a.ReferencedSOPInstanceUID for a in graphic_annotation.ReferencedImageSequence]
        if target_sop_uid is None or target_sop_uid in referenced_uids:
            # A TextObjectSequence may be present but no GraphicObjectSequence so ".get" is used
            graphic_items = list(gen_graphic_items(graphic_annotation.get("GraphicObjectSequence", [])))
            for graphic_item in group_polylines(graphic_items):
                yield Annotation(sop_uids=referenced_uids, data=graphic_item.data, form=graphic_item.form)


def get_pr_reference_targets(dcm: Dicom) -> Optional[List[str]]:
    targets = [uid for annotation in dcm_to_annotations(dcm) for uid in annotation.uids]
    return targets if targets else None


def dcms_to_annotations(dcms: List[Dicom]) -> Iterator[Annotation]:
    for dcm in [d for d in dcms if d.Modality == presentation_modality]:
        try:
            yield from dcm_to_annotations(dcm)
        except Exception as e:
            logger.info(e)


def overlay_annotations(images: List[DicomImage], annotations: List[Annotation]) -> None:
    for image in images:
        for annotation in annotations:
            if image.uid in annotation.uids:
                if annotation.is_rectangle:
                    draw_bbox(image.pixels, annotation.trace)
                else:
                    raise Exception("Drawing for non-rectangular traces is not currently supported.")


def to_rgb(image: ndarray) -> ndarray:
    chns, rows, cols = image.shape
    rgb = np.zeros((chns, 3, rows, cols))

    for i, channel in enumerate(image):
        norm_pixels = channel / channel.max()

        for j in range(3):
            rgb[i, j, :, :] = norm_pixels

    return rgb


def dcms_to_annotated_images(dcms: List[Dicom]) -> List[DicomImage]:
    images = list(dcms_to_images(dcms))
    annotations = list(dcms_to_annotations(dcms))
    overlay_annotations(images, annotations)
    return images


def to_8bit(x: ndarray) -> ndarray:
    min, max = x.min(), x.max()
    x = (x - min) / (max - min) * 255
    return x.astype(np.uint8)

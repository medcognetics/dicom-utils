from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Iterator, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np
from numpy import ndarray
from pydicom.uid import UID
from tqdm import tqdm

from .dicom import read_dicom_image
from .logging import logger
from .types import Dicom, DicomAttributeSequence


Color = Tuple[int, int, int]
Bbox = Tuple[int, int, int, int]
presentation_modality: Final[str] = "PR"


def kwargs2str(**kwargs: Any) -> str:
    key_val_str = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in kwargs.items())
    return f"<{key_val_str}>"


class Form(Enum):
    CIRCLE = "CIRCLE"
    ELLIPSE = "ELLIPSE"
    POLYLINE = "POLYLINE"
    # POINT - defined by DICOM standard but not currently supported
    # INTERPOLATED - defined by DICOM standard but not currently supported

    def __init__(self, value) -> None:
        # Make sure CIRCLE is not assigned the name "CIRCL" or some other misspelling
        assert self.value == self.name == value, f"Enum name/value mismatch {self.value} != {self.name}"


@dataclass
class Reference:
    uid: UID
    frame: int  # The first frame is denoted as frame number 1

    @classmethod
    def from_graphic_annotation(cls, ann: Dicom) -> "Reference":
        return cls(UID(ann.ReferencedSOPInstanceUID), int(ann.get("ReferencedFrameNumber", 1)))

    def __repr__(self):
        return kwargs2str(SOPInstanceUID=self.uid, frame=self.frame)


@dataclass
class Annotation:
    """Store an annotation with corresponding DICOM filename"""

    def __init__(self, refs: List[Reference], data: List[float], form: Form) -> None:
        self.refs: List[Reference] = refs
        self.form: Form = Form(form)
        self.trace = dicom_trace_to_bbox(data, self.form)
        self.is_rectangle: bool = True  # TODO Add non-rectangular trace support

    @property
    def uids(self) -> List[UID]:
        return [ref.uid for ref in self.refs]

    def __repr__(self):
        return kwargs2str(references=self.refs, form=self.form, trace=self.trace, is_rectangle=self.is_rectangle)


@dataclass
class DicomImage:
    """Store DICOM image pixels with associated metadata"""

    pixels: ndarray
    uid: UID

    @classmethod
    def from_dicom(cls, dicom: Dicom, **kwargs) -> "DicomImage":
        pixels = to_rgb(read_dicom_image(dicom, **kwargs))
        return cls(pixels, dicom.SOPInstanceUID)

    @property
    def is_single_frame(self) -> bool:
        return self.pixels.shape[0] == 1

    def __repr__(self):
        return kwargs2str(SOPInstanceUID=self.uid, shape=self.pixels.shape)


class GraphicItem(NamedTuple):
    """Store info from DICOM graphic objects. More information about DICOM graphic objects can be found here:
    http://dicom.nema.org/medical/Dicom/2017c/output/chtml/part03/sect_C.10.5.html
    """

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

    dtype = images[0].dtype
    collage_rows = 1 if num_images < 3 else 2
    collage_cols = int(num_images / collage_rows + 0.5)

    assert all(i.shape[1] == 3 for i in images)
    collage = np.zeros((image_chns, 3, collage_rows * max_image_rows, collage_cols * max_image_cols), dtype=dtype)

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


def draw_rectangle(
    image: ndarray,
    coord0: Tuple[int, int],
    coord1: Tuple[int, int],
    color: Color = (0, 255, 0),
    thickness: int = 30,
) -> ndarray:
    # OpenCV can give weird errors without the following line
    image = cv2.UMat(image).get()  # type: ignore
    cv2.rectangle(image, coord0, coord1, color, thickness)
    return image


def dcms_to_images(dcms: List[Dicom], bar: bool = True, jobs: int = 4, **kwargs) -> Iterator[DicomImage]:
    """Some DICOMs may only contain annotations or other non-image information, so we want to
    pull out image data where applicable."""
    tqdm_bar = tqdm(desc="Loading images", total=len(dcms), disable=(not bar))

    def func(dcm: Dicom) -> Optional[DicomImage]:
        try:
            return DicomImage.from_dicom(dcm, **kwargs)
        except Exception as e:
            logger.info(e)

    def callback(f: Future):
        tqdm_bar.update(1)

    with ThreadPoolExecutor(jobs) as tp:
        futures: List[Future] = [tp.submit(func, dcm) for dcm in dcms]
        for f in futures:
            f.add_done_callback(callback)

        for f in futures:
            if result := f.result():
                yield result
    tqdm_bar.close()


def distance(a: List[float], b: List[float]) -> float:
    assert len(a) == len(b), "Expected lists of equal length for calculating distance"
    return sum((u - v) ** 2 for u, v in zip(a, b))


def are_contiguous_points(a: List[float], b: List[float]) -> bool:
    a_start = a[:2]
    a_stop = a[-2:]
    b_start = b[:2]
    return distance(a_start, a_stop) > distance(a_stop, b_start)


def are_contiguous_polylines(a: GraphicItem, b: GraphicItem) -> bool:
    return a.form == b.form == Form.POLYLINE and are_contiguous_points(a.data, b.data)


def group_polylines(graphic_items: List[GraphicItem]) -> Iterator[GraphicItem]:
    """Consecutive polyline traces may have been recorded separately when they were intended to be part of one single
    trace. Identify this situation and combine polylines accordingly."""
    graphic_items = graphic_items.copy()
    while graphic_items:
        item = graphic_items.pop(0)

        while graphic_items and are_contiguous_polylines(item, graphic_items[0]):
            item += graphic_items.pop(0)

        yield item


def gen_graphic_items(graphic_objects: DicomAttributeSequence) -> Iterator[GraphicItem]:
    """DICOM graphic objects contain info we don't really care about right now,
    so we distill the info interest into a generator of GraphicItems."""
    for graphic_object in graphic_objects:
        assert graphic_object.GraphicAnnotationUnits == "PIXEL"
        assert graphic_object.GraphicDimensions == 2
        yield GraphicItem(data=graphic_object.GraphicData, form=Form(graphic_object.GraphicType))


def dcm_to_annotations(dcm: Dicom, target_sop_uid: Optional[UID] = None) -> Iterator[Annotation]:
    """Search through a DICOM for graphic annotations, and only return annotations corresponding to
    a specific SOPInstanceUID if desired."""
    for graphic_annotation in dcm.get("GraphicAnnotationSequence", []):
        refs = [Reference.from_graphic_annotation(a) for a in graphic_annotation.ReferencedImageSequence]
        if target_sop_uid is None or target_sop_uid in [ref.uid for ref in refs]:
            # A TextObjectSequence may be present but no GraphicObjectSequence so ".get" is used
            graphic_items = list(gen_graphic_items(graphic_annotation.get("GraphicObjectSequence", [])))
            for graphic_item in group_polylines(graphic_items):
                yield Annotation(refs=refs, data=graphic_item.data, form=graphic_item.form)


def get_pr_reference_targets(dcm: Dicom) -> Optional[List[UID]]:
    targets = [uid for annotation in dcm_to_annotations(dcm) for uid in annotation.uids]
    return targets if targets else None


def dcms_to_annotations(dcms: List[Dicom], bar: bool = True) -> Iterator[Annotation]:
    dcms = [d for d in dcms if d.get("Modality", "") == presentation_modality]
    for dcm in tqdm(dcms, desc="Loading annotations", disable=(not bar), leave=bool(dcms)):
        try:
            yield from dcm_to_annotations(dcm)
        except Exception as e:
            logger.info(e)


def overlay_annotations_on_image(image: DicomImage, annotations: List[Annotation]) -> None:
    pixels = image.pixels
    assert pixels.shape[1] == 3, "Drawing assumes 3 color channels"
    assert pixels.dtype == np.uint8, "Drawing assumes 8-bits per color channel"

    for i, frame in enumerate(pixels):
        frame = chw_to_hwc(frame)

        for annotation in annotations:
            # The annotation is currently drawn on every frame, but we could selectively draw on only
            # the applicable frames by checking frames specified in annotation.refs
            assert annotation.is_rectangle, "Drawing for non-rectangular traces is not currently supported."
            x0, y0, x1, y1 = annotation.trace
            frame = draw_rectangle(frame, (x0, y0), (x1, y1))

        pixels[i] = hwc_to_chw(frame)


def overlay_annotations(images: List[DicomImage], annotations: List[Annotation]) -> None:
    for image in images:
        overlay_annotations_on_image(image, [ann for ann in annotations if image.uid in ann.uids])


def to_rgb(image: ndarray) -> ndarray:
    # 2D images will come as 1xHxW (treat as 1xHxW) and 3D images will come as 1xDxHxW (treat as DxHxW)
    assert image.shape[0] == 1, ""
    chns, rows, cols = image.shape[-3:]
    image = image.reshape(chns, rows, cols)

    # uint8 is sufficient for visualization and saves memory which is important for 3D images
    rgb = np.zeros((chns, 3, rows, cols), dtype=np.uint8)

    for i, channel in enumerate(image):
        for j in range(3):
            rgb[i, j, :, :] = to_8bit(channel)

    return rgb


def dcms_to_annotated_images(dcms: List[Dicom], **kwargs) -> List[DicomImage]:
    images = list(dcms_to_images(dcms, **kwargs))
    annotations = list(dcms_to_annotations(dcms))
    overlay_annotations(images, annotations)
    return images


def to_8bit(x: ndarray) -> ndarray:
    pixel_min, pixel_max = x.min(), x.max()
    delta = max(pixel_max - pixel_min, 1e-9)
    x = (x - pixel_min) / delta * 255
    return x.round().astype(np.uint8)

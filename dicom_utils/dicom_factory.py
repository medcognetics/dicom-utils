#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pydicom
from pathlib import Path

class DicomFactory:
    ...

    def __init__(self):
        series_uid = pydicom.uid.generate_uid()
        study_uid = pydicom.uid.generate_uid()

    @property
    def image(self):
        ...

    @image.getter
    def image(self):
        ...

    @property
    def save(self, path: Path) -> None:
        ...



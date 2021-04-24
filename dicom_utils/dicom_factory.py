#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import pydicom


class DicomFactory:
    ...

    def __init__(self):
        pydicom.uid.generate_uid()
        pydicom.uid.generate_uid()

    @property
    def image(self):
        ...

    @image.getter
    def image(self):
        ...

    @property
    def save(self, path: Path) -> None:
        ...

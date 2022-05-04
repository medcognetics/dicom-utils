#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Union

from pydicom.uid import UID


SeriesUID = UID
StudyUID = UID
SOPUID = UID
TransferSyntaxUID = UID
ImageUID = Union[SeriesUID, SOPUID]

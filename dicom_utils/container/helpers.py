#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Union


SeriesUID = str
StudyUID = str
SOPUID = str
TransferSyntaxUID = str
ImageUID = Union[SeriesUID, SOPUID]

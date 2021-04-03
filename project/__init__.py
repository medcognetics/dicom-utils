#!/usr/bin/env python
# -*- coding: utf-8 -*-
try:
    from .version import __version__
except ImportError:
    __version__ = "Unknown"

__all__ = [__version__]

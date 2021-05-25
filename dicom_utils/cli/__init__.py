#!/usr/bin/env python
# -*- coding: utf-8 -*-
from signal import SIG_DFL, SIGINT, SIGPIPE, signal


# avoid problems w/ keyboard interrupt when pipeing
signal(SIGPIPE, SIG_DFL)
signal(SIGINT, SIG_DFL)

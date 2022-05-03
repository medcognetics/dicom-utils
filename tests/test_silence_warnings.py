#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib as imp
import os
import warnings

import pytest

import dicom_utils
from dicom_utils import SILENCE_ENV_VAR, SPAM_WARNING_PATTERNS, filter_spam_warnings


@pytest.mark.parametrize("pattern", SPAM_WARNING_PATTERNS)
def test_filtered_patterns(mocker, pattern):
    # NOTE: pytest.warns seems to ignore filters, so test using mock calls instead
    spy = mocker.spy(warnings, "filterwarnings")
    filter_spam_warnings()
    return any(call.args == ("ignore", pattern) for call in spy.mock_calls)


@pytest.mark.parametrize("value", ["", "0", "1"])
def test_ignore_spam_env_var(mocker, value):
    # NOTE: mock on filter_spam_warnings() would be reset during import, so use filterwarnings
    os.environ[SILENCE_ENV_VAR] = value
    spy = mocker.spy(warnings, "filterwarnings")
    imp.reload(dicom_utils)
    ignored_all_spam = all(any(call.args == ("ignore", p) for call in spy.mock_calls) for p in SPAM_WARNING_PATTERNS)
    should_ignore_all_spam = value == "1"
    assert ignored_all_spam == should_ignore_all_spam

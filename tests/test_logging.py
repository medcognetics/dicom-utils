import logging

import pydicom
import pytest

from dicom_utils.logging import LoggingLevel, logger, set_logging_level


@pytest.mark.parametrize(
    "logging_level, pydicom_logging_level, expected_logging_level, expected_pydicom_logging_level",
    [
        (None, None, LoggingLevel.WARNING, LoggingLevel.WARNING),
        (LoggingLevel.DEBUG, None, LoggingLevel.DEBUG, LoggingLevel.DEBUG),
        (None, LoggingLevel.ERROR, LoggingLevel.WARNING, LoggingLevel.ERROR),
        (LoggingLevel.INFO, LoggingLevel.CRITICAL, LoggingLevel.INFO, LoggingLevel.CRITICAL),
    ],
)
def test_set_logging_level(
    logging_level, pydicom_logging_level, expected_logging_level, expected_pydicom_logging_level
):
    set_logging_level(logging_level, pydicom_logging_level)
    assert logging.getLevelName(logger.level) == expected_logging_level.value
    assert logging.getLevelName(pydicom.config.logger.level) == expected_pydicom_logging_level.value

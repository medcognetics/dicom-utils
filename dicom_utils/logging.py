import logging
from enum import Enum
from typing import Optional

import pydicom


logging.basicConfig()
logger = logging.getLogger("dicom-utils")


class ExtendedEnum(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class LoggingLevel(ExtendedEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def set_logging_level(logging_level: Optional[LoggingLevel], pydicom_logging_level: Optional[LoggingLevel]) -> None:
    """
    Set logging levels for this package.
    If pydicom_logging_level is None, logging_level will override pydicom_logging_level.

    Args:
        logging_level:
            Logging level for this package which may override the pydicom logging level
        pydicom_logging_level:
            Logging level for pydicom

    Returns:
        None
    """
    logging_level = LoggingLevel(logging_level or LoggingLevel.WARNING)
    pydicom_logging_level = LoggingLevel(pydicom_logging_level or logging_level)

    logger.setLevel(logging_level.value)
    pydicom.config.logger.setLevel(pydicom_logging_level.value)

    logger.debug(f"Package logging set to {logging_level} and pydicom logging set to {pydicom_logging_level}.")

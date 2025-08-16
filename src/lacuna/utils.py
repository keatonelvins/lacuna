"""Logging and miscellaneous utilities."""

import logging


def setup_logger(name: str = "lacuna") -> logging.Logger:
    """Setup logger."""
    logger = logging.getLogger(name)
    return logger
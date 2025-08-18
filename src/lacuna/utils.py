"""Logging and miscellaneous utilities."""

from loguru import logger

from .distributed import get_rank


def _rank_filter(record):
    """Filter to only log from rank 0."""
    return get_rank() == 0


def setup_logger() -> None:
    """Configure loguru with colored output and timestamps."""
    logger.remove()  # Remove default handler
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
        level="INFO",
        filter=_rank_filter,
    )

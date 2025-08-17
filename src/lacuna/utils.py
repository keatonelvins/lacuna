"""Logging and miscellaneous utilities."""

from loguru import logger


def setup_logger() -> None:
    """Configure loguru with colored output and timestamps."""
    logger.remove()  # Remove default handler
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
        level="INFO",
    )

"""Logging configuration for VMC Trading Bot."""

import sys
from pathlib import Path
from loguru import logger

# Remove default handler
logger.remove()

def setup_logger(
    log_level: str = "INFO",
    log_file: str | None = None,
    rotation: str = "10 MB",
    retention: str = "7 days"
) -> None:
    """
    Configure the logger for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, logs to console only.
        rotation: When to rotate log files
        retention: How long to keep old log files
    """
    # Console handler with colored output
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip"
        )

def get_logger(name: str = "vmc_bot"):
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name for identification

    Returns:
        Logger instance
    """
    return logger.bind(name=name)

"""
Logging configuration for the ConvFinQA pipeline.

Wires loguru sinks at application startup. Call configure_logging() once from
the entry point before any other module emits log messages.
"""

from datetime import UTC, datetime

from loguru import logger

_LOG_FILE = "/code/logs/convfinqa.log"
_LOG_ROTATION = "50 MB"
_LOG_RETENTION = 5
_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} - {message}"


def configure_logging() -> None:
    """Add a rotating file sink to loguru and write a run-separator banner.

    The default stderr sink is kept. The file sink rotates at 50 MB and retains
    the last 5 files so logs do not grow without bound. A separator banner is
    written at the start of each run so individual runs are clearly partitioned
    when reviewing the log file.
    """
    logger.add(
        _LOG_FILE,
        level="DEBUG",
        rotation=_LOG_ROTATION,
        retention=_LOG_RETENTION,
        encoding="utf-8",
        format=_LOG_FORMAT,
    )
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 60}\n  NEW RUN — {timestamp}\n{'=' * 60}\n\n")
    logger.info(f"Logging to file: {_LOG_FILE}")

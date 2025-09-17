import logging

from src.config import config


def get_logger(name: str = __name__) -> logging.Logger:
    """Create a logger based on configuration.

    Args:
        name (str): Name of the logger. Defaults to the module's name.

    Returns:
        logging.Logger: Configured logger instance.

    """
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        logger.setLevel(getattr(logging, config.log_level, logging.INFO))

        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_format)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        logger.propagate = False

    return logger

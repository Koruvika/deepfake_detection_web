"""Custom logger"""

import logging

def configure_logging():
    """Initialize logging defaults for Project.

    This function does:
    - Assign INFO and DEBUG level to logger file handler and console handler.

    Returns:
        Logger.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create a file handler with a lower log level
    file_handler = logging.FileHandler('logger.log')
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and add it to the handlers
    default_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] "
        "[%(funcName)s():%(lineno)s] %(message)s",
        "%d/%m/%Y %H:%M:%S")

    file_handler.setFormatter(default_formatter)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(file_handler)

    return logger

custom_logger = configure_logging()

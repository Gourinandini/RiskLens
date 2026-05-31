"""
Logging Configuration Module.

Provides a unified logger configuration with a standardized format:
timestamp | level | module | message.
"""

import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Configure and return a logger with a unified formatting style.

    Args:
        name (str): The name of the logger, typically __name__.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if the logger has already been initialized
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Standard format: timestamp | level | module | message
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger

import logging
import os
from logging.handlers import RotatingFileHandler
from config.settings import STORAGE_SETTINGS, LOGGING_SETTINGS

def setup_logger():
    """Set up logging with file and console output."""
    logger = logging.getLogger("voice_to_voice")
    logger.setLevel(getattr(logging, LOGGING_SETTINGS['level']))

    # Avoid duplicate handlers
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(LOGGING_SETTINGS['format']))
        logger.addHandler(console_handler)

        # File handler with rotation
        os.makedirs(os.path.dirname(LOGGING_SETTINGS['file']), exist_ok=True)
        file_handler = RotatingFileHandler(
            LOGGING_SETTINGS['file'],
            maxBytes=LOGGING_SETTINGS['max_size'],
            backupCount=LOGGING_SETTINGS['backup_count']
        )
        file_handler.setFormatter(logging.Formatter(LOGGING_SETTINGS['format']))
        logger.addHandler(file_handler)

    return logger
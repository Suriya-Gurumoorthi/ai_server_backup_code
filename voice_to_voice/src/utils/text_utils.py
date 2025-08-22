from src.utils.logger import setup_logger

def clean_text(text):
    """Clean text input."""
    logger = setup_logger()
    if text:
        cleaned = text.strip().lower()
        logger.info(f"Cleaned text: {cleaned}")
        return cleaned
    return ""
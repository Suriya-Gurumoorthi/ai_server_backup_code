import scipy.io.wavfile as wavfile
from src.utils.logger import setup_logger

def save_audio(data, output_path, sample_rate):
    """Save audio data to a file."""
    logger = setup_logger()
    try:
        wavfile.write(output_path, sample_rate, data)
        logger.info(f"Saved audio to {output_path}")
    except Exception as e:
        logger.error(f"Error saving audio: {e}")
        raise
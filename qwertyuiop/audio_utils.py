"""
Audio utilities for handling different audio formats and conversions.
Provides functions to validate, convert, and process audio data.
"""

import io
import wave
import logging
import numpy as np
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)

def is_valid_wav(audio_bytes: bytes) -> bool:
    """
    Check if audio bytes contain a valid WAV file header.
    
    Args:
        audio_bytes: Raw audio data
        
    Returns:
        True if valid WAV format, False otherwise
    """
    if len(audio_bytes) < 12:
        return False
    
    # Check for RIFF header
    if audio_bytes[:4] != b'RIFF':
        return False
    
    # Check for WAVE format
    if audio_bytes[8:12] != b'WAVE':
        return False
    
    return True

def wrap_pcm_in_wav(pcm_bytes: bytes, sample_rate: int = 16000, 
                    sample_width: int = 2, channels: int = 1) -> bytes:
    """
    Wrap raw PCM audio data in a WAV header.
    
    Args:
        pcm_bytes: Raw PCM audio data
        sample_rate: Sample rate in Hz
        sample_width: Sample width in bytes (2 for 16-bit)
        channels: Number of channels (1 for mono)
        
    Returns:
        WAV-formatted audio data
    """
    try:
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Error wrapping PCM in WAV: {e}")
        return pcm_bytes

def convert_audio_to_wav(audio_bytes: bytes, sample_rate: int = 16000, 
                        sample_width: int = 2, channels: int = 1) -> bytes:
    """
    Convert audio data to WAV format if it's not already.
    
    Args:
        audio_bytes: Raw audio data
        sample_rate: Sample rate in Hz
        sample_width: Sample width in bytes
        channels: Number of channels
        
    Returns:
        WAV-formatted audio data
    """
    # Check if already valid WAV
    if is_valid_wav(audio_bytes):
        logger.debug("Audio is already in WAV format")
        return audio_bytes
    
    # Assume raw PCM and wrap in WAV header
    logger.info(f"Converting raw PCM to WAV format: {len(audio_bytes)} bytes")
    return wrap_pcm_in_wav(audio_bytes, sample_rate, sample_width, channels)

def validate_audio_format(audio_bytes: bytes) -> Tuple[bool, str]:
    """
    Validate audio format and return status information.
    
    Args:
        audio_bytes: Raw audio data
        
    Returns:
        Tuple of (is_valid, format_info)
    """
    if not audio_bytes:
        return False, "Empty audio data"
    
    if len(audio_bytes) < 12:
        return False, f"Audio data too short: {len(audio_bytes)} bytes"
    
    # Check for WAV format
    if is_valid_wav(audio_bytes):
        return True, "Valid WAV format"
    
    # Check for common raw PCM indicators
    if len(audio_bytes) % 2 == 0:  # Even number of bytes (16-bit samples)
        return True, "Raw PCM format (16-bit)"
    
    return False, f"Unknown format: {audio_bytes[:16].hex()}"

def get_audio_info(audio_bytes: bytes) -> dict:
    """
    Get information about audio data.
    
    Args:
        audio_bytes: Raw audio data
        
    Returns:
        Dictionary with audio information
    """
    info = {
        'length_bytes': len(audio_bytes),
        'is_wav': is_valid_wav(audio_bytes),
        'first_16_bytes_hex': audio_bytes[:16].hex() if len(audio_bytes) >= 16 else audio_bytes.hex(),
        'estimated_samples': len(audio_bytes) // 2,  # Assuming 16-bit
        'estimated_duration_ms': (len(audio_bytes) // 2) / 16000 * 1000  # Assuming 16kHz
    }
    
    if is_valid_wav(audio_bytes):
        try:
            # Try to read WAV header information
            with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
                info.update({
                    'channels': wf.getnchannels(),
                    'sample_width': wf.getsampwidth(),
                    'sample_rate': wf.getframerate(),
                    'frames': wf.getnframes(),
                    'duration_seconds': wf.getnframes() / wf.getframerate()
                })
        except Exception as e:
            info['wav_read_error'] = str(e)
    
    return info

def safe_audio_conversion(audio_bytes: bytes, target_sample_rate: int = 16000) -> bytes:
    """
    Safely convert audio to the target format, handling various input formats.
    
    Args:
        audio_bytes: Raw audio data
        target_sample_rate: Target sample rate
        
    Returns:
        Converted audio data in WAV format
    """
    try:
        # Validate input
        is_valid, format_info = validate_audio_format(audio_bytes)
        logger.info(f"Audio validation: {is_valid}, {format_info}")
        
        if not is_valid:
            logger.error(f"Invalid audio format: {format_info}")
            return audio_bytes
        
        # Convert to WAV if needed
        if not is_valid_wav(audio_bytes):
            logger.info("Converting raw PCM to WAV format")
            return convert_audio_to_wav(audio_bytes, target_sample_rate)
        
        logger.debug("Audio is already in WAV format, no conversion needed")
        return audio_bytes
        
    except Exception as e:
        logger.error(f"Error in safe audio conversion: {e}")
        return audio_bytes

def debug_audio_bytes(audio_bytes: bytes, prefix: str = "Audio") -> None:
    """
    Log debug information about audio bytes.
    
    Args:
        audio_bytes: Raw audio data
        prefix: Prefix for log messages
    """
    if not audio_bytes:
        logger.debug(f"{prefix}: Empty audio data")
        return
    
    info = get_audio_info(audio_bytes)
    logger.debug(f"{prefix} Info:")
    logger.debug(f"  Length: {info['length_bytes']} bytes")
    logger.debug(f"  Is WAV: {info['is_wav']}")
    logger.debug(f"  First 16 bytes: {info['first_16_bytes_hex']}")
    logger.debug(f"  Estimated samples: {info['estimated_samples']}")
    logger.debug(f"  Estimated duration: {info['estimated_duration_ms']:.1f}ms")
    
    if 'sample_rate' in info:
        logger.debug(f"  Sample rate: {info['sample_rate']}Hz")
        logger.debug(f"  Channels: {info['channels']}")
        logger.debug(f"  Sample width: {info['sample_width']} bytes")
        logger.debug(f"  Duration: {info['duration_seconds']:.2f}s")

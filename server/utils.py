"""
Utility functions and logging configuration for the Ultravox WebSocket server.
"""

import logging
import asyncio
import websockets
import numpy as np
from typing import Any, Dict, Tuple


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


async def safe_send_response(websocket, message, client_address):
    """Safely send a response, handling connection state properly."""
    try:
        await websocket.send(message)
        logger = logging.getLogger(__name__)
        logger.info(f"Response sent successfully to {client_address}")
        return True
    except websockets.ConnectionClosed:
        logger = logging.getLogger(__name__)
        logger.warning(f"Connection to {client_address} closed while sending response")
        return False
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error sending response to {client_address}: {e}")
        return False


def get_connection_info(websocket) -> str:
    """Get formatted client address information."""
    return f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"


def validate_text_for_tts(text: str) -> str:
    """Validate and clean text for TTS processing."""
    if not text or not text.strip():
        return ""
    
    # Truncate very long text
    if len(text) > 1000:
        return text[:1000]
    
    return text.strip()


def is_valid_speech_audio(audio_bytes: bytes, 
                          min_energy_threshold: int = 500,
                          min_speech_ratio: float = 0.1,
                          sample_rate: int = 16000) -> Tuple[bool, dict]:
    """
    Validate if audio bytes contain actual speech vs noise/silence.
    
    Args:
        audio_bytes: Raw 16-bit PCM audio data
        min_energy_threshold: Minimum RMS energy for speech
        min_speech_ratio: Minimum ratio of speech-like samples (0.0-1.0)
        sample_rate: Audio sample rate in Hz
        
    Returns:
        (is_valid, stats): Tuple of validation result and audio statistics
    """
    try:
        if not audio_bytes or len(audio_bytes) < 2:
            return False, {"reason": "empty_audio"}
        
        # Ensure even number of bytes (16-bit samples)
        if len(audio_bytes) % 2 != 0:
            audio_bytes = audio_bytes + b'\x00'
        
        # Convert to numpy array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Calculate audio statistics
        rms_energy = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
        max_amplitude = np.max(np.abs(audio_array))
        mean_amplitude = np.mean(np.abs(audio_array))
        
        # Calculate zero-crossing rate (speech has more zero crossings)
        zero_crossings = np.sum(np.diff(np.sign(audio_array)) != 0)
        zcr = zero_crossings / len(audio_array) if len(audio_array) > 0 else 0
        
        # Calculate percentage of non-zero samples
        nonzero_samples = np.count_nonzero(audio_array)
        nonzero_ratio = nonzero_samples / len(audio_array)
        
        # Calculate speech-like sample ratio (samples above threshold)
        speech_threshold = min_energy_threshold * 0.5  # Half of RMS threshold
        speech_samples = np.sum(np.abs(audio_array) > speech_threshold)
        speech_ratio = speech_samples / len(audio_array)
        
        # Collect statistics
        stats = {
            "rms_energy": float(rms_energy),
            "max_amplitude": int(max_amplitude),
            "mean_amplitude": float(mean_amplitude),
            "zcr": float(zcr),
            "nonzero_ratio": float(nonzero_ratio),
            "speech_ratio": float(speech_ratio),
            "duration_ms": (len(audio_array) / sample_rate) * 1000,
            "samples": len(audio_array)
        }
        
        # Validation criteria
        has_sufficient_energy = rms_energy > min_energy_threshold
        has_sufficient_activity = zcr > 0.01  # At least 1% zero crossings
        has_sufficient_speech = speech_ratio > min_speech_ratio
        is_not_mostly_zeros = nonzero_ratio > 0.2  # At least 20% non-zero
        
        is_valid = (has_sufficient_energy and 
                   has_sufficient_activity and 
                   has_sufficient_speech and 
                   is_not_mostly_zeros)
        
        # Add validation results to stats
        stats["is_valid"] = is_valid
        stats["validation_details"] = {
            "has_sufficient_energy": has_sufficient_energy,
            "has_sufficient_activity": has_sufficient_activity,
            "has_sufficient_speech": has_sufficient_speech,
            "is_not_mostly_zeros": is_not_mostly_zeros
        }
        
        if not is_valid:
            # Determine primary reason for rejection
            if not has_sufficient_energy:
                stats["reason"] = "low_energy"
            elif not has_sufficient_activity:
                stats["reason"] = "low_activity"
            elif not has_sufficient_speech:
                stats["reason"] = "insufficient_speech"
            elif not is_not_mostly_zeros:
                stats["reason"] = "mostly_silence"
            else:
                stats["reason"] = "unknown"
        else:
            stats["reason"] = "valid_speech"
        
        logger = logging.getLogger(__name__)
        logger.debug(f"Audio validation: is_valid={is_valid}, reason={stats['reason']}, "
                    f"rms={rms_energy:.1f}, zcr={zcr:.3f}, speech_ratio={speech_ratio:.3f}")
        
        return is_valid, stats
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error validating audio quality: {e}")
        return False, {"reason": "validation_error", "error": str(e)}


def should_transcribe_audio(audio_bytes: bytes, 
                            min_energy: int = 800,  # Higher threshold than VAD
                            min_speech_ratio: float = 0.15) -> bool:
    """
    Simplified check specifically for transcription decision.
    Uses higher thresholds than VAD to avoid false positives.
    
    Args:
        audio_bytes: Raw audio data
        min_energy: Minimum RMS energy (higher = stricter)
        min_speech_ratio: Minimum ratio of speech samples
        
    Returns:
        True if audio should be transcribed, False otherwise
    """
    is_valid, stats = is_valid_speech_audio(
        audio_bytes, 
        min_energy_threshold=min_energy,
        min_speech_ratio=min_speech_ratio
    )
    
    if not is_valid:
        logger = logging.getLogger(__name__)
        logger.info(f"Skipping transcription: {stats['reason']} "
                   f"(energy={stats.get('rms_energy', 0):.1f}, "
                   f"speech_ratio={stats.get('speech_ratio', 0):.3f})")
    
    return is_valid


def unified_audio_validation(audio_bytes: bytes, 
                           min_energy: int = 1500,  # Use same strict threshold as config
                           min_speech_ratio: float = 0.25) -> bool:
    """
    Unified audio validation for both Whisper and Ultravox processing.
    Uses the same strict criteria to ensure consistent behavior.
    
    Args:
        audio_bytes: Raw audio data
        min_energy: Minimum RMS energy (same as TRANSCRIPTION_MIN_ENERGY)
        min_speech_ratio: Minimum ratio of speech samples (same as TRANSCRIPTION_MIN_SPEECH_RATIO)
        
    Returns:
        True if audio should be processed by both Whisper and Ultravox, False otherwise
    """
    is_valid, stats = is_valid_speech_audio(
        audio_bytes, 
        min_energy_threshold=min_energy,
        min_speech_ratio=min_speech_ratio
    )
    
    if not is_valid:
        logger = logging.getLogger(__name__)
        logger.info(f"[UNIFIED] Rejecting audio processing: {stats['reason']} "
                   f"(energy={stats.get('rms_energy', 0):.1f}, "
                   f"speech_ratio={stats.get('speech_ratio', 0):.3f})")
    else:
        logger = logging.getLogger(__name__)
        logger.info(f"[UNIFIED] Audio passed validation: "
                   f"energy={stats.get('rms_energy', 0):.1f}, "
                   f"speech_ratio={stats.get('speech_ratio', 0):.3f}")
    
    return is_valid

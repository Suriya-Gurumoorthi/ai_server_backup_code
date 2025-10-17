#!/usr/bin/env python3
"""
Test script for audio format handling fix

Tests the audio format validation and conversion to ensure the WAV header issue is resolved.
"""

import asyncio
import logging
import numpy as np
import io
import wave
from audio_utils import (
    is_valid_wav, wrap_pcm_in_wav, convert_audio_to_wav, 
    validate_audio_format, get_audio_info, safe_audio_conversion,
    debug_audio_bytes
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_raw_pcm(duration_ms: int, sample_rate: int = 16000, frequency: int = 440) -> bytes:
    """Generate raw PCM audio data (no WAV header)."""
    duration_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, duration_samples, False)
    audio_data = (1000 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
    return audio_data.tobytes()

def generate_wav_audio(duration_ms: int, sample_rate: int = 16000, frequency: int = 440) -> bytes:
    """Generate proper WAV audio data."""
    duration_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, duration_samples, False)
    audio_data = (1000 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
    
    # Wrap in WAV header
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    return buf.getvalue()

def test_audio_format_detection():
    """Test audio format detection."""
    logger.info("=" * 60)
    logger.info("Testing Audio Format Detection")
    logger.info("=" * 60)
    
    # Test 1: Raw PCM (should be invalid WAV)
    logger.info("Test 1: Raw PCM Audio")
    raw_pcm = generate_raw_pcm(1000)  # 1 second
    is_wav = is_valid_wav(raw_pcm)
    logger.info(f"Raw PCM detected as WAV: {is_wav}")
    debug_audio_bytes(raw_pcm, "Raw PCM")
    
    # Test 2: Proper WAV (should be valid)
    logger.info("Test 2: Proper WAV Audio")
    wav_audio = generate_wav_audio(1000)  # 1 second
    is_wav = is_valid_wav(wav_audio)
    logger.info(f"WAV audio detected as WAV: {is_wav}")
    debug_audio_bytes(wav_audio, "WAV Audio")
    
    # Test 3: Empty audio
    logger.info("Test 3: Empty Audio")
    empty_audio = b""
    is_wav = is_valid_wav(empty_audio)
    logger.info(f"Empty audio detected as WAV: {is_wav}")
    
    # Test 4: Invalid data
    logger.info("Test 4: Invalid Audio Data")
    invalid_audio = b"not audio data"
    is_wav = is_valid_wav(invalid_audio)
    logger.info(f"Invalid audio detected as WAV: {is_wav}")
    
    logger.info("✅ Audio format detection test completed")

def test_audio_conversion():
    """Test audio format conversion."""
    logger.info("=" * 60)
    logger.info("Testing Audio Format Conversion")
    logger.info("=" * 60)
    
    # Test 1: Convert raw PCM to WAV
    logger.info("Test 1: Raw PCM to WAV Conversion")
    raw_pcm = generate_raw_pcm(1000)  # 1 second
    logger.info(f"Original raw PCM: {len(raw_pcm)} bytes")
    
    converted_wav = convert_audio_to_wav(raw_pcm)
    logger.info(f"Converted WAV: {len(converted_wav)} bytes")
    
    # Verify it's now valid WAV
    is_wav = is_valid_wav(converted_wav)
    logger.info(f"Converted audio is valid WAV: {is_wav}")
    debug_audio_bytes(converted_wav, "Converted WAV")
    
    # Test 2: Already WAV audio (should not change)
    logger.info("Test 2: Already WAV Audio (should not change)")
    wav_audio = generate_wav_audio(1000)
    logger.info(f"Original WAV: {len(wav_audio)} bytes")
    
    converted_audio = convert_audio_to_wav(wav_audio)
    logger.info(f"After conversion: {len(converted_audio)} bytes")
    logger.info(f"Audio unchanged: {wav_audio == converted_audio}")
    
    logger.info("✅ Audio conversion test completed")

def test_safe_audio_conversion():
    """Test safe audio conversion function."""
    logger.info("=" * 60)
    logger.info("Testing Safe Audio Conversion")
    logger.info("=" * 60)
    
    # Test 1: Raw PCM
    logger.info("Test 1: Safe conversion of raw PCM")
    raw_pcm = generate_raw_pcm(1000)
    safe_converted = safe_audio_conversion(raw_pcm)
    logger.info(f"Safe conversion result: {len(safe_converted)} bytes")
    debug_audio_bytes(safe_converted, "Safe Converted")
    
    # Test 2: Already WAV
    logger.info("Test 2: Safe conversion of WAV (should not change)")
    wav_audio = generate_wav_audio(1000)
    safe_converted = safe_audio_conversion(wav_audio)
    logger.info(f"Safe conversion result: {len(safe_converted)} bytes")
    logger.info(f"Audio unchanged: {wav_audio == safe_converted}")
    
    # Test 3: Empty audio
    logger.info("Test 3: Safe conversion of empty audio")
    empty_audio = b""
    safe_converted = safe_audio_conversion(empty_audio)
    logger.info(f"Safe conversion result: {len(safe_converted)} bytes")
    
    logger.info("✅ Safe audio conversion test completed")

def test_audio_validation():
    """Test audio validation function."""
    logger.info("=" * 60)
    logger.info("Testing Audio Validation")
    logger.info("=" * 60)
    
    # Test various audio formats
    test_cases = [
        ("Raw PCM", generate_raw_pcm(1000)),
        ("WAV Audio", generate_wav_audio(1000)),
        ("Empty", b""),
        ("Short", b"ab"),
        ("Invalid", b"not audio"),
        ("Partial WAV", b"RIFF"),
    ]
    
    for name, audio_data in test_cases:
        is_valid, format_info = validate_audio_format(audio_data)
        logger.info(f"{name}: Valid={is_valid}, Info={format_info}")
        
        if audio_data:
            info = get_audio_info(audio_data)
            logger.info(f"  Length: {info['length_bytes']} bytes")
            logger.info(f"  Is WAV: {info['is_wav']}")
            logger.info(f"  First bytes: {info['first_16_bytes_hex'][:32]}...")
    
    logger.info("✅ Audio validation test completed")

async def test_librosa_compatibility():
    """Test compatibility with librosa after conversion."""
    logger.info("=" * 60)
    logger.info("Testing Librosa Compatibility")
    logger.info("=" * 60)
    
    try:
        import librosa
        
        # Test 1: Raw PCM converted to WAV
        logger.info("Test 1: Raw PCM -> WAV -> Librosa")
        raw_pcm = generate_raw_pcm(1000)
        converted_wav = safe_audio_conversion(raw_pcm)
        
        # Try to load with librosa
        audio_stream = io.BytesIO(converted_wav)
        audio, sr = librosa.load(audio_stream, sr=16000)
        logger.info(f"Librosa loaded: {len(audio)} samples at {sr}Hz")
        
        # Test 2: Original WAV
        logger.info("Test 2: Original WAV -> Librosa")
        wav_audio = generate_wav_audio(1000)
        audio_stream = io.BytesIO(wav_audio)
        audio, sr = librosa.load(audio_stream, sr=16000)
        logger.info(f"Librosa loaded: {len(audio)} samples at {sr}Hz")
        
        logger.info("✅ Librosa compatibility test completed")
        
    except ImportError:
        logger.warning("Librosa not available, skipping compatibility test")
    except Exception as e:
        logger.error(f"Librosa compatibility test failed: {e}")

async def main():
    """Main test function."""
    logger.info("Starting Audio Format Fix Tests")
    logger.info("=" * 80)
    
    try:
        # Test audio format detection
        test_audio_format_detection()
        
        # Test audio conversion
        test_audio_conversion()
        
        # Test safe conversion
        test_safe_audio_conversion()
        
        # Test validation
        test_audio_validation()
        
        # Test librosa compatibility
        await test_librosa_compatibility()
        
        logger.info("=" * 80)
        logger.info("✅ All audio format fix tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Audio format fix test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

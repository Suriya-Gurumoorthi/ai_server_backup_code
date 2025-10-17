#!/usr/bin/env python3
"""
Test script for VAD integration in Ultravox Server

Tests the Voice Activity Detection system to ensure it works correctly.
"""

import asyncio
import logging
import numpy as np
import wave
import io
from typing import Optional

# Import VAD components
from voice_activity_detection import VoiceActivityDetector
from vad_manager import vad_manager
from config import (
    VAD_ENABLED, VAD_ENERGY_THRESHOLD, VAD_SILENCE_DURATION_MS, 
    VAD_MIN_SPEECH_DURATION_MS, VAD_HIGH_PASS_CUTOFF, 
    VAD_MIN_CONSECUTIVE_FRAMES, VAD_SPECTRAL_FLATNESS_THRESHOLD,
    AUDIO_SAMPLE_RATE
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_test_audio(duration_ms: int, frequency: int = 440, amplitude: int = 1000) -> bytes:
    """Generate test audio signal."""
    sample_rate = AUDIO_SAMPLE_RATE
    duration_samples = int(sample_rate * duration_ms / 1000)
    
    # Generate sine wave
    t = np.linspace(0, duration_ms / 1000, duration_samples, False)
    audio_data = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
    
    return audio_data.tobytes()

def generate_silence(duration_ms: int) -> bytes:
    """Generate silence audio."""
    sample_rate = AUDIO_SAMPLE_RATE
    duration_samples = int(sample_rate * duration_ms / 1000)
    return b'\x00' * (duration_samples * 2)  # 16-bit samples

def generate_noise(duration_ms: int, amplitude: int = 100) -> bytes:
    """Generate noise audio."""
    sample_rate = AUDIO_SAMPLE_RATE
    duration_samples = int(sample_rate * duration_ms / 1000)
    
    # Generate white noise
    noise = np.random.normal(0, amplitude, duration_samples).astype(np.int16)
    return noise.tobytes()

def save_audio_debug(filename: str, audio_data: bytes, sample_rate: int = AUDIO_SAMPLE_RATE):
    """Save audio data as WAV file for debugging."""
    try:
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
        logger.info(f"Saved audio debug file: {filename}")
    except Exception as e:
        logger.error(f"Failed to save audio debug file {filename}: {e}")

async def test_vad_basic_functionality():
    """Test basic VAD functionality."""
    logger.info("=" * 60)
    logger.info("Testing VAD Basic Functionality")
    logger.info("=" * 60)
    
    # Create VAD instance
    vad = VoiceActivityDetector(
        energy_threshold=VAD_ENERGY_THRESHOLD,
        silence_duration_ms=VAD_SILENCE_DURATION_MS,
        min_speech_duration_ms=VAD_MIN_SPEECH_DURATION_MS,
        sample_rate=AUDIO_SAMPLE_RATE,
        high_pass_cutoff=VAD_HIGH_PASS_CUTOFF,
        min_consecutive_frames=VAD_MIN_CONSECUTIVE_FRAMES,
        spectral_flatness_threshold=VAD_SPECTRAL_FLATNESS_THRESHOLD,
        debug_logging=True
    )
    
    # Test 1: Speech detection
    logger.info("Test 1: Speech Detection")
    speech_audio = generate_test_audio(1000, 440, 2000)  # 1 second, 440Hz, loud
    is_speech = vad.is_speech(speech_audio)
    logger.info(f"Speech audio detected as speech: {is_speech}")
    save_audio_debug("test_speech.wav", speech_audio)
    
    # Test 2: Silence detection
    logger.info("Test 2: Silence Detection")
    silence_audio = generate_silence(1000)  # 1 second silence
    is_silence = vad.is_speech(silence_audio)
    logger.info(f"Silence audio detected as speech: {is_silence}")
    save_audio_debug("test_silence.wav", silence_audio)
    
    # Test 3: Noise detection
    logger.info("Test 3: Noise Detection")
    noise_audio = generate_noise(1000, 50)  # 1 second low-level noise
    is_noise = vad.is_speech(noise_audio)
    logger.info(f"Noise audio detected as speech: {is_noise}")
    save_audio_debug("test_noise.wav", noise_audio)
    
    # Test 4: Barge-in detection
    logger.info("Test 4: Barge-in Detection")
    vad.barge_in_consecutive_frames_threshold = 2
    barge_in_audio = generate_test_audio(200, 440, 3000)  # Short, loud burst
    is_barge_in = vad.is_speech_for_barge_in(barge_in_audio)
    logger.info(f"Barge-in audio detected: {is_barge_in}")
    save_audio_debug("test_barge_in.wav", barge_in_audio)
    
    # Test 5: Audio chunk processing
    logger.info("Test 5: Audio Chunk Processing")
    vad2 = VoiceActivityDetector(
        energy_threshold=VAD_ENERGY_THRESHOLD,
        silence_duration_ms=VAD_SILENCE_DURATION_MS,
        min_speech_duration_ms=VAD_MIN_SPEECH_DURATION_MS,
        sample_rate=AUDIO_SAMPLE_RATE,
        high_pass_cutoff=VAD_HIGH_PASS_CUTOFF,
        min_consecutive_frames=VAD_MIN_CONSECUTIVE_FRAMES,
        spectral_flatness_threshold=VAD_SPECTRAL_FLATNESS_THRESHOLD,
        debug_logging=True
    )
    
    # Simulate speech followed by silence
    speech_chunk = generate_test_audio(500, 440, 2000)
    silence_chunk = generate_silence(600)  # Longer than silence_duration_ms
    
    # Process speech chunk
    speech_segment = vad2.process_audio_chunk(speech_chunk)
    logger.info(f"Speech chunk processing result: {speech_segment is not None}")
    
    # Process silence chunk
    speech_segment = vad2.process_audio_chunk(silence_chunk)
    logger.info(f"Silence chunk processing result: {speech_segment is not None}")
    
    if speech_segment:
        logger.info(f"Complete speech segment detected: {len(speech_segment)} bytes")
        save_audio_debug("test_complete_speech.wav", speech_segment)
    
    # Get VAD statistics
    stats = vad.get_stats()
    logger.info(f"VAD Statistics: {stats}")
    
    logger.info("✅ Basic VAD functionality test completed")

async def test_vad_manager():
    """Test VAD manager functionality."""
    logger.info("=" * 60)
    logger.info("Testing VAD Manager")
    logger.info("=" * 60)
    
    # Test connection management
    connection_id = "test_connection_123"
    
    # Create VAD for connection
    vad_instance = vad_manager.create_vad_for_connection(connection_id)
    logger.info(f"VAD instance created: {vad_instance is not None}")
    
    # Test audio processing
    test_audio = generate_test_audio(1000, 440, 2000)
    processed_audio = vad_manager.process_audio_chunk(connection_id, test_audio)
    logger.info(f"Audio processing result: {processed_audio is not None}")
    
    # Test barge-in detection
    barge_in_audio = generate_test_audio(200, 440, 3000)
    is_barge_in = vad_manager.detect_barge_in(connection_id, barge_in_audio)
    logger.info(f"Barge-in detection: {is_barge_in}")
    
    # Test speech detection
    is_speech = vad_manager.is_speech(connection_id, test_audio)
    logger.info(f"Speech detection: {is_speech}")
    
    # Get connection stats
    connection_stats = vad_manager.get_connection_stats(connection_id)
    logger.info(f"Connection stats: {connection_stats}")
    
    # Get global stats
    global_stats = vad_manager.get_global_stats()
    logger.info(f"Global stats: {global_stats}")
    
    # Remove VAD for connection
    vad_manager.remove_vad_for_connection(connection_id)
    logger.info("VAD instance removed")
    
    logger.info("✅ VAD Manager test completed")

async def test_vad_integration():
    """Test VAD integration with server components."""
    logger.info("=" * 60)
    logger.info("Testing VAD Integration")
    logger.info("=" * 60)
    
    # Test configuration
    logger.info(f"VAD Enabled: {VAD_ENABLED}")
    logger.info(f"Energy Threshold: {VAD_ENERGY_THRESHOLD}")
    logger.info(f"Silence Duration: {VAD_SILENCE_DURATION_MS}ms")
    logger.info(f"Min Speech Duration: {VAD_MIN_SPEECH_DURATION_MS}ms")
    logger.info(f"High-pass Cutoff: {VAD_HIGH_PASS_CUTOFF}Hz")
    logger.info(f"Consecutive Frames: {VAD_MIN_CONSECUTIVE_FRAMES}")
    logger.info(f"Spectral Flatness Threshold: {VAD_SPECTRAL_FLATNESS_THRESHOLD}")
    
    # Test VAD manager initialization
    logger.info(f"VAD Manager Enabled: {vad_manager.enabled}")
    logger.info(f"VAD Manager Stats: {vad_manager.get_global_stats()}")
    
    logger.info("✅ VAD Integration test completed")

async def main():
    """Main test function."""
    logger.info("Starting VAD Integration Tests")
    logger.info("=" * 80)
    
    try:
        # Test basic VAD functionality
        await test_vad_basic_functionality()
        
        # Test VAD manager
        await test_vad_manager()
        
        # Test integration
        await test_vad_integration()
        
        logger.info("=" * 80)
        logger.info("✅ All VAD tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ VAD test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

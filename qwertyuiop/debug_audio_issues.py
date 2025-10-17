#!/usr/bin/env python3
"""
Comprehensive Audio Issues Diagnostic Script
Tests all components of the audio pipeline to identify problems.
"""

import asyncio
import logging
import numpy as np
import io
import wave
from typing import Optional, Tuple
import sys
import os

# Add server directory to path
sys.path.append('/home/novel/server')

from config import (
    VAD_ENABLED, VAD_ENERGY_THRESHOLD, VAD_SILENCE_DURATION_MS, 
    VAD_MIN_SPEECH_DURATION_MS, VAD_HIGH_PASS_CUTOFF, 
    VAD_MIN_CONSECUTIVE_FRAMES, VAD_SPECTRAL_FLATNESS_THRESHOLD,
    AUDIO_SAMPLE_RATE, TTS_SAMPLE_RATE
)
from voice_activity_detection import VoiceActivityDetector
from audio_utils import validate_audio_format, debug_audio_bytes, safe_audio_conversion
from models import model_manager
from audio_processor import audio_processor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def generate_test_audio(duration_ms: int = 1000, sample_rate: int = 16000, 
                       frequency: int = 440, amplitude: int = 1000) -> bytes:
    """Generate test audio with specified parameters."""
    duration_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, duration_samples, False)
    audio_data = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
    return audio_data.tobytes()

def generate_wav_audio(duration_ms: int = 1000, sample_rate: int = 16000, 
                      frequency: int = 440, amplitude: int = 1000) -> bytes:
    """Generate proper WAV audio data."""
    duration_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, duration_samples, False)
    audio_data = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
    
    # Wrap in WAV header
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    return buf.getvalue()

def test_vad_sensitivity():
    """Test VAD sensitivity with different audio levels."""
    logger.info("=" * 60)
    logger.info("Testing VAD Sensitivity")
    logger.info("=" * 60)
    
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
    
    # Test different audio amplitudes
    amplitudes = [100, 500, 1000, 2000, 5000, 10000]
    
    for amplitude in amplitudes:
        audio_data = generate_test_audio(1000, AUDIO_SAMPLE_RATE, 440, amplitude)
        is_speech = vad.is_speech(audio_data)
        
        # Calculate RMS energy
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        rms_energy = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
        
        logger.info(f"Amplitude {amplitude:5d}: RMS={rms_energy:8.1f}, Speech={is_speech}")
    
    logger.info("✅ VAD sensitivity test completed")

def test_audio_format_handling():
    """Test audio format handling in the pipeline."""
    logger.info("=" * 60)
    logger.info("Testing Audio Format Handling")
    logger.info("=" * 60)
    
    # Test 1: Raw PCM audio
    logger.info("Test 1: Raw PCM Audio")
    raw_pcm = generate_test_audio(1000, AUDIO_SAMPLE_RATE, 440, 1000)
    is_valid, format_info = validate_audio_format(raw_pcm)
    logger.info(f"Raw PCM validation: {is_valid}, {format_info}")
    
    # Convert to WAV
    converted_wav = safe_audio_conversion(raw_pcm)
    is_valid_wav, format_info_wav = validate_audio_format(converted_wav)
    logger.info(f"Converted WAV validation: {is_valid_wav}, {format_info_wav}")
    
    # Test 2: Already WAV audio
    logger.info("Test 2: WAV Audio")
    wav_audio = generate_wav_audio(1000, AUDIO_SAMPLE_RATE, 440, 1000)
    is_valid, format_info = validate_audio_format(wav_audio)
    logger.info(f"WAV validation: {is_valid}, {format_info}")
    
    # Test 3: Low amplitude audio (might be filtered by VAD)
    logger.info("Test 3: Low Amplitude Audio")
    low_amp_audio = generate_test_audio(1000, AUDIO_SAMPLE_RATE, 440, 100)
    is_valid, format_info = validate_audio_format(low_amp_audio)
    logger.info(f"Low amplitude validation: {is_valid}, {format_info}")
    
    logger.info("✅ Audio format handling test completed")

def test_vad_processing():
    """Test VAD processing with different audio types."""
    logger.info("=" * 60)
    logger.info("Testing VAD Processing")
    logger.info("=" * 60)
    
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
    
    # Test different audio types
    test_cases = [
        ("Normal Speech", generate_test_audio(1000, AUDIO_SAMPLE_RATE, 440, 1000)),
        ("Low Volume", generate_test_audio(1000, AUDIO_SAMPLE_RATE, 440, 200)),
        ("High Frequency", generate_test_audio(1000, AUDIO_SAMPLE_RATE, 2000, 1000)),
        ("Low Frequency", generate_test_audio(1000, AUDIO_SAMPLE_RATE, 100, 1000)),
        ("Silence", b'\x00' * 32000),  # 1 second of silence
        ("Noise", np.random.randint(-100, 100, 16000, dtype=np.int16).tobytes())
    ]
    
    for name, audio_data in test_cases:
        logger.info(f"Testing {name}:")
        
        # Test speech detection
        is_speech = vad.is_speech(audio_data)
        logger.info(f"  Speech detected: {is_speech}")
        
        # Test chunk processing
        speech_segment = vad.process_audio_chunk(audio_data)
        if speech_segment:
            logger.info(f"  Speech segment returned: {len(speech_segment)} bytes")
        else:
            logger.info(f"  No speech segment returned")
        
        # Test barge-in detection
        is_barge_in = vad.is_speech_for_barge_in(audio_data)
        logger.info(f"  Barge-in detected: {is_barge_in}")
        
        logger.info("")
    
    logger.info("✅ VAD processing test completed")

async def test_tts_generation():
    """Test TTS audio generation."""
    logger.info("=" * 60)
    logger.info("Testing TTS Generation")
    logger.info("=" * 60)
    
    try:
        # Test TTS availability
        tts_available = model_manager.is_tts_available()
        logger.info(f"TTS available: {tts_available}")
        
        if not tts_available:
            logger.warning("TTS not available - skipping TTS tests")
            return
        
        # Test TTS generation
        test_text = "Hello, this is a test of the text to speech system."
        logger.info(f"Generating TTS for: '{test_text}'")
        
        tts_audio = await audio_processor.generate_tts_audio(test_text)
        
        if tts_audio:
            logger.info(f"TTS audio generated: {len(tts_audio)} bytes")
            
            # Validate the generated audio
            is_valid, format_info = validate_audio_format(tts_audio)
            logger.info(f"TTS audio validation: {is_valid}, {format_info}")
            
            # Debug audio info
            debug_audio_bytes(tts_audio, "TTS Generated Audio")
        else:
            logger.error("TTS audio generation failed")
        
        logger.info("✅ TTS generation test completed")
        
    except Exception as e:
        logger.error(f"TTS test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

def test_websocket_audio_flow():
    """Test the complete WebSocket audio flow."""
    logger.info("=" * 60)
    logger.info("Testing WebSocket Audio Flow")
    logger.info("=" * 60)
    
    # Simulate the audio flow from WebSocket handler
    logger.info("Simulating WebSocket audio processing...")
    
    # Generate test audio
    test_audio = generate_wav_audio(1000, AUDIO_SAMPLE_RATE, 440, 1000)
    logger.info(f"Generated test audio: {len(test_audio)} bytes")
    
    # Step 1: Audio format validation
    is_valid, format_info = validate_audio_format(test_audio)
    logger.info(f"Step 1 - Format validation: {is_valid}, {format_info}")
    
    if not is_valid:
        logger.error("Audio format validation failed - this would cause issues")
        return
    
    # Step 2: VAD processing (if enabled)
    if VAD_ENABLED:
        logger.info("Step 2 - VAD processing (enabled)")
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
        
        speech_segment = vad.process_audio_chunk(test_audio)
        if speech_segment:
            logger.info(f"VAD detected speech: {len(speech_segment)} bytes")
        else:
            logger.warning("VAD filtered out audio - this could be the problem!")
            logger.warning("Try lowering VAD_ENERGY_THRESHOLD or VAD_MIN_CONSECUTIVE_FRAMES")
    else:
        logger.info("Step 2 - VAD processing (disabled)")
    
    # Step 3: Audio conversion
    logger.info("Step 3 - Audio conversion")
    converted_audio = safe_audio_conversion(test_audio)
    logger.info(f"Converted audio: {len(converted_audio)} bytes")
    
    logger.info("✅ WebSocket audio flow test completed")

def print_configuration_issues():
    """Print potential configuration issues."""
    logger.info("=" * 60)
    logger.info("Configuration Analysis")
    logger.info("=" * 60)
    
    logger.info("Current VAD Configuration:")
    logger.info(f"  VAD_ENABLED: {VAD_ENABLED}")
    logger.info(f"  VAD_ENERGY_THRESHOLD: {VAD_ENERGY_THRESHOLD}")
    logger.info(f"  VAD_SILENCE_DURATION_MS: {VAD_SILENCE_DURATION_MS}")
    logger.info(f"  VAD_MIN_SPEECH_DURATION_MS: {VAD_MIN_SPEECH_DURATION_MS}")
    logger.info(f"  VAD_HIGH_PASS_CUTOFF: {VAD_HIGH_PASS_CUTOFF}")
    logger.info(f"  VAD_MIN_CONSECUTIVE_FRAMES: {VAD_MIN_CONSECUTIVE_FRAMES}")
    logger.info(f"  VAD_SPECTRAL_FLATNESS_THRESHOLD: {VAD_SPECTRAL_FLATNESS_THRESHOLD}")
    
    logger.info("\nPotential Issues:")
    
    if VAD_ENERGY_THRESHOLD > 1000:
        logger.warning(f"  ⚠️  VAD_ENERGY_THRESHOLD ({VAD_ENERGY_THRESHOLD}) might be too high")
        logger.warning("     Try lowering to 200-500 for better sensitivity")
    
    if VAD_MIN_CONSECUTIVE_FRAMES > 2:
        logger.warning(f"  ⚠️  VAD_MIN_CONSECUTIVE_FRAMES ({VAD_MIN_CONSECUTIVE_FRAMES}) might be too high")
        logger.warning("     Try lowering to 1-2 for more responsive detection")
    
    if VAD_HIGH_PASS_CUTOFF > 300:
        logger.warning(f"  ⚠️  VAD_HIGH_PASS_CUTOFF ({VAD_HIGH_PASS_CUTOFF}) might filter out important frequencies")
        logger.warning("     Try lowering to 100-200 Hz")
    
    if VAD_SPECTRAL_FLATNESS_THRESHOLD < 0.5:
        logger.warning(f"  ⚠️  VAD_SPECTRAL_FLATNESS_THRESHOLD ({VAD_SPECTRAL_FLATNESS_THRESHOLD}) might be too restrictive")
        logger.warning("     Try increasing to 0.7-0.9")
    
    logger.info("✅ Configuration analysis completed")

async def main():
    """Main diagnostic function."""
    logger.info("🔍 Starting Comprehensive Audio Issues Diagnostic")
    logger.info("=" * 80)
    
    try:
        # Test configuration issues
        print_configuration_issues()
        
        # Test audio format handling
        test_audio_format_handling()
        
        # Test VAD sensitivity
        test_vad_sensitivity()
        
        # Test VAD processing
        test_vad_processing()
        
        # Test WebSocket audio flow
        test_websocket_audio_flow()
        
        # Test TTS generation
        await test_tts_generation()
        
        logger.info("=" * 80)
        logger.info("🎯 DIAGNOSTIC SUMMARY")
        logger.info("=" * 80)
        logger.info("If you're not hearing audio, check these common issues:")
        logger.info("1. VAD is filtering out your audio (most common)")
        logger.info("2. Audio format incompatibility")
        logger.info("3. TTS generation failure")
        logger.info("4. WebSocket connection issues")
        logger.info("5. Audio encoding/decoding problems")
        logger.info("")
        logger.info("💡 RECOMMENDED FIXES:")
        logger.info("- Lower VAD_ENERGY_THRESHOLD to 200-300")
        logger.info("- Set VAD_MIN_CONSECUTIVE_FRAMES to 1-2")
        logger.info("- Disable VAD temporarily: VAD_ENABLED = False")
        logger.info("- Check WebSocket client audio format")
        logger.info("- Verify TTS model is loaded correctly")
        
    except Exception as e:
        logger.error(f"❌ Diagnostic failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main())

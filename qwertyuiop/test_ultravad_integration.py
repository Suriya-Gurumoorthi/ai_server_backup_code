#!/usr/bin/env python3
"""
Test script for ultraVAD integration.
Tests the ultraVAD model loading and basic functionality.
"""

import sys
import os
import logging
import numpy as np
import librosa

# Add the server directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultravad_manager import ultravad_manager

def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_ultravad_loading():
    """Test ultraVAD model loading."""
    logger = setup_logging()
    logger.info("Testing ultraVAD model loading...")
    
    # Test model loading
    success = ultravad_manager.load_model()
    if success:
        logger.info("✅ ultraVAD model loaded successfully")
        return True
    else:
        logger.error("❌ ultraVAD model failed to load")
        return False

def test_ultravad_detection():
    """Test ultraVAD interruption detection with dummy audio."""
    logger = setup_logging()
    logger.info("Testing ultraVAD interruption detection...")
    
    if not ultravad_manager.is_model_available():
        logger.error("❌ ultraVAD model not available for testing")
        return False
    
    try:
        # Generate dummy audio data (1 second of silence)
        sample_rate = 16000
        duration = 1.0  # 1 second
        audio_array = np.zeros(int(sample_rate * duration), dtype=np.float32)
        
        # Convert to bytes (simulate audio bytes)
        audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()
        
        # Test conversation turns
        conversation_turns = [
            {"role": "assistant", "content": "Hi, how are you?"},
            {"role": "user", "content": "I'm doing well, thank you."}
        ]
        
        # Test interruption detection
        is_interruption, confidence = ultravad_manager.detect_interruption(
            audio_bytes, conversation_turns
        )
        
        logger.info(f"Detection result: is_interruption={is_interruption}, confidence={confidence:.6f}")
        
        # Test threshold setting
        ultravad_manager.set_threshold(0.2)
        new_threshold = ultravad_manager.get_threshold()
        logger.info(f"Threshold test: set=0.2, got={new_threshold}")
        
        logger.info("✅ ultraVAD detection test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ ultraVAD detection test failed: {e}")
        return False

def test_ultravad_shutdown():
    """Test ultraVAD manager shutdown."""
    logger = setup_logging()
    logger.info("Testing ultraVAD manager shutdown...")
    
    try:
        ultravad_manager.shutdown()
        logger.info("✅ ultraVAD manager shutdown completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ ultraVAD manager shutdown failed: {e}")
        return False

def main():
    """Run all ultraVAD integration tests."""
    logger = setup_logging()
    logger.info("🚀 Starting ultraVAD integration tests...")
    
    tests = [
        ("Model Loading", test_ultravad_loading),
        ("Interruption Detection", test_ultravad_detection),
        ("Manager Shutdown", test_ultravad_shutdown)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} test FAILED with exception: {e}")
    
    logger.info(f"\n🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All ultraVAD integration tests PASSED!")
        return 0
    else:
        logger.error("💥 Some ultraVAD integration tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())


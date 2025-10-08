"""
Main entry point for Vicidial Bridge

This module provides the main entry point and logging setup for the modular voice bot system.
"""

import asyncio
import logging
import os
import sys
from typing import Optional

try:
    from .config import (
        PRODUCTION_MODE, PRODUCTION_LOG_LEVEL, VICIAL_AUDIOSOCKET_HOST, VICIAL_AUDIOSOCKET_PORT,
        LOCAL_MODEL_WS_URL, TARGET_EXTENSION, VICIAL_SAMPLE_RATE, LOCAL_MODEL_SAMPLE_RATE,
        AUDIO_PROCESS_INTERVAL, VAD_ENABLED, SAVE_AI_AUDIO, SAVE_DEBUG_AUDIO, SAVE_COMPLETE_CALLS,
        LOCAL_MODEL_CORPUS_ID, RAG_MAX_RESULTS, RAG_MIN_SCORE
    )
    from .bridge import VicialLocalModelBridge
except ImportError:
    from config import (
        PRODUCTION_MODE, PRODUCTION_LOG_LEVEL, VICIAL_AUDIOSOCKET_HOST, VICIAL_AUDIOSOCKET_PORT,
        LOCAL_MODEL_WS_URL, TARGET_EXTENSION, VICIAL_SAMPLE_RATE, LOCAL_MODEL_SAMPLE_RATE,
        AUDIO_PROCESS_INTERVAL, VAD_ENABLED, SAVE_AI_AUDIO, SAVE_DEBUG_AUDIO, SAVE_COMPLETE_CALLS,
        LOCAL_MODEL_CORPUS_ID, RAG_MAX_RESULTS, RAG_MIN_SCORE
    )
    from bridge import VicialLocalModelBridge

logger: Optional[logging.Logger] = None

def setup_logging():
    """Configure comprehensive logging for the bridge"""
    global logger
    
    # Create logs directory if it doesn't exist
    log_dir = "/var/log/vicial_local_model"
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging with multiple handlers
    logging.basicConfig(
        level=PRODUCTION_LOG_LEVEL,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(f'{log_dir}/vicial_local_model_bridge.log')  # Main log
        ]
    )
    
    # Add error-only handler separately
    error_handler = logging.FileHandler(f'{log_dir}/vicial_local_model_errors.log')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | %(message)s'))
    logging.getLogger().addHandler(error_handler)
    
    # Create logger
    logger = logging.getLogger(__name__)
    
    # Log startup information
    logger.info("=" * 80)
    logger.info("VICIAL-LOCAL MODEL VOICE BRIDGE STARTING")
    logger.info("=" * 80)
    logger.info(f"Vicial AudioSocket: {VICIAL_AUDIOSOCKET_HOST}:{VICIAL_AUDIOSOCKET_PORT}")
    logger.info(f"Local Model Server: {LOCAL_MODEL_WS_URL}")
    logger.info(f"Target Extension: {TARGET_EXTENSION}")
    logger.info(f"Audio: {VICIAL_SAMPLE_RATE}Hz -> {LOCAL_MODEL_SAMPLE_RATE}Hz (x{LOCAL_MODEL_SAMPLE_RATE // VICIAL_SAMPLE_RATE})")
    logger.info(f"LATENCY OPTIMIZATION: {AUDIO_PROCESS_INTERVAL*1000:.0f}ms processing interval (was 200ms)")
    logger.info(f"Audio Buffering: {AUDIO_PROCESS_INTERVAL*1000:.0f}ms minimum, {AUDIO_PROCESS_INTERVAL*1000*2:.0f}ms maximum")
    logger.info(f"VOICE ACTIVITY DETECTION: {'ENABLED' if VAD_ENABLED else 'DISABLED'}")
    if VAD_ENABLED:
        try:
            from .config import VAD_ENERGY_THRESHOLD, VAD_SILENCE_DURATION_MS, VAD_MIN_SPEECH_DURATION_MS, VAD_DEBUG_LOGGING
        except ImportError:
            from config import VAD_ENERGY_THRESHOLD, VAD_SILENCE_DURATION_MS, VAD_MIN_SPEECH_DURATION_MS, VAD_DEBUG_LOGGING
        logger.info(f"VAD Parameters: energy_threshold={VAD_ENERGY_THRESHOLD}, silence_duration={VAD_SILENCE_DURATION_MS}ms, min_speech={VAD_MIN_SPEECH_DURATION_MS}ms")
        logger.info(f"VAD Debug Logging: {'ENABLED' if VAD_DEBUG_LOGGING else 'DISABLED'}")
    if SAVE_AI_AUDIO:
        try:
            from .config import AI_AUDIO_DIR
        except ImportError:
            from config import AI_AUDIO_DIR
        logger.info(f"AI Audio Saving: ENABLED -> {AI_AUDIO_DIR}")
        os.makedirs(AI_AUDIO_DIR, exist_ok=True)
    else:
        logger.info("AI Audio Saving: DISABLED")
    
    if SAVE_DEBUG_AUDIO:
        try:
            from .config import DEBUG_AUDIO_DIR
        except ImportError:
            from config import DEBUG_AUDIO_DIR
        logger.info(f"DEBUG Audio Saving: ENABLED -> {DEBUG_AUDIO_DIR}")
        os.makedirs(DEBUG_AUDIO_DIR, exist_ok=True)
    else:
        logger.info("DEBUG Audio Saving: DISABLED")
    
    if SAVE_COMPLETE_CALLS:
        try:
            from .config import CALL_RECORDINGS_DIR, SAVE_SEPARATE_TRACKS, SAVE_CALL_METADATA
        except ImportError:
            from config import CALL_RECORDINGS_DIR, SAVE_SEPARATE_TRACKS, SAVE_CALL_METADATA
        logger.info(f"COMPLETE Call Recording: ENABLED -> {CALL_RECORDINGS_DIR}")
        logger.info(f"  - Separate Tracks: {'ENABLED' if SAVE_SEPARATE_TRACKS else 'DISABLED'}")
        logger.info(f"  - Metadata Files: {'ENABLED' if SAVE_CALL_METADATA else 'DISABLED'}")
        os.makedirs(CALL_RECORDINGS_DIR, exist_ok=True)
    else:
        logger.info("COMPLETE Call Recording: DISABLED")
    logger.info("AI Processing: Local model server via WebSocket")
    logger.info("TTS Engine: Piper neural voices (high-quality)")
    logger.info("Audio Chunks: Real-time streaming | WebSocket binary messages")
    logger.info(f"RAG Enabled: YES -> Corpus ID: {LOCAL_MODEL_CORPUS_ID[:8]}...")
    logger.info(f"RAG Parameters: max_results={RAG_MAX_RESULTS}, min_score={RAG_MIN_SCORE}")
    logger.info(f"Production Mode: {PRODUCTION_MODE} (Log Level: {PRODUCTION_LOG_LEVEL})")
    logger.info("=" * 80)
    
    return logger

async def main():
    """Main entry point for the Vicial-Local Model bridge"""
    try:
        # Setup logging first
        setup_logging()
        
        # Create and start the bridge
        bridge = VicialLocalModelBridge()
        await bridge.start_server()
    except KeyboardInterrupt:
        if logger:
            logger.info("Bridge stopped by user (Ctrl+C)")
    except Exception as e:
        if logger:
            logger.error(f"Fatal error in bridge: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        else:
            print(f"Fatal error in bridge: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")

def run_bridge():
    """Convenience function to run the bridge"""
    try:
        # Use asyncio.run() for Python 3.7+, fallback to loop.run_until_complete() for older versions
        asyncio.run(main())
    except KeyboardInterrupt:
        if logger:
            logger.info("Bridge interrupted by user")
        else:
            print("Bridge interrupted by user")
    except Exception as e:
        if logger:
            logger.error(f"Error running bridge: {e}")
        else:
            print(f"Error running bridge: {e}")

if __name__ == "__main__":
    run_bridge()

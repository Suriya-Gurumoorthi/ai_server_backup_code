"""
Vicidial Bridge - Modular Voice Bot for Local Model Integration

This package provides a bridge between Vicial AudioSocket and local AI model servers,
enabling real-time voice conversations with advanced features like Voice Activity Detection,
call recording, and high-quality TTS.
"""

__version__ = "1.0.0"
__author__ = "Vicidial Bridge Team"

# Import main components for easy access
from .bridge import VicialLocalModelBridge
from .config import *

__all__ = [
    'VicialLocalModelBridge',
    'VICIAL_AUDIOSOCKET_HOST',
    'VICIAL_AUDIOSOCKET_PORT',
    'LOCAL_MODEL_HOST',
    'LOCAL_MODEL_PORT',
    'LOCAL_MODEL_WS_URL',
    'TARGET_EXTENSION',
    'VAD_ENABLED',
    'SAVE_COMPLETE_CALLS',
    'SAVE_AI_AUDIO',
    'SAVE_DEBUG_AUDIO'
]

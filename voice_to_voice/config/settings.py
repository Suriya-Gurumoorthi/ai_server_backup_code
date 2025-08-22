"""
Configuration settings for the Voice-to-Voice AI System
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Audio settings
AUDIO_SETTINGS = {
    'sample_rate': 16000,
    'channels': 1,
    'chunk_size': 1024,
    'format': 'int16',
    'device_index': None,  # Auto-detect
    'silence_threshold': 0.01,
    'silence_duration': 1.0,  # seconds
    'max_audio_length': 30.0,  # seconds
}

# Piper TTS settings
PIPER_SETTINGS = {
    'model_path': str(BASE_DIR / 'models' / 'en_US-lessac-medium.onnx'),
    'config_path': str(BASE_DIR / 'models' / 'en_US-lessac-medium.onnx.json'),
    'voice_quality': 'medium',
    'speed': 1.0,
    'noise_scale': 0.667,
    'length_scale': 1.0,
    'noise_w': 0.8,
}

# Ultravox STT settings
ULTRAVOX_SETTINGS = {
    'model_path': str(BASE_DIR / 'models' / 'ultravox_model'),
    'language': 'en',
    'beam_size': 5,
    'best_of': 5,
    'temperature': 0.0,
    'compression_ratio_threshold': 2.4,
    'log_prob_threshold': -1.0,
    'no_speech_threshold': 0.6,
    'condition_on_previous_text': True,
    'initial_prompt': None,
}

# Conversation settings
CONVERSATION_SETTINGS = {
    'max_turns': 50,
    'timeout': 300,  # seconds
    'interruption_threshold': 0.3,
    'context_window': 10,  # number of turns to remember
    'role_prompt_template': "You are a {role}. {description}",
}

# Interruption settings
INTERRUPTION_SETTINGS = {
    'detection_threshold': 0.5,
    'response_delay': 0.1,  # seconds
    'grace_period': 0.5,  # seconds
}

# Web interface settings
WEB_SETTINGS = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
    'secret_key': os.getenv('SECRET_KEY', 'your-secret-key-here'),
    'websocket_ping_interval': 25,
    'websocket_ping_timeout': 10,
}

# Data storage settings
STORAGE_SETTINGS = {
    'audio_dir': str(BASE_DIR / 'data' / 'audio'),
    'conversations_dir': str(BASE_DIR / 'data' / 'conversations'),
    'reports_dir': str(BASE_DIR / 'data' / 'reports'),
    'models_dir': str(BASE_DIR / 'models'),
    'temp_dir': str(BASE_DIR / 'temp'),
}

# Logging settings
LOGGING_SETTINGS = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': str(BASE_DIR / 'logs' / 'voice_system.log'),
    'max_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
}

# AI Model settings
AI_MODEL_SETTINGS = {
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
    'max_tokens': 1000,
    'temperature': 0.7,
    'top_p': 0.9,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0,
}

# Call termination keywords
TERMINATION_KEYWORDS = [
    'goodbye', 'bye', 'end call', 'hang up', 'terminate',
    'stop', 'exit', 'quit', 'finish', 'done'
]

# Environment variables
ENV_VARS = {
    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
    'ULTRAVOX_API_KEY': os.getenv('ULTRAVOX_API_KEY'),
    'PIPER_MODEL_PATH': os.getenv('PIPER_MODEL_PATH'),
    'DEBUG': os.getenv('DEBUG', 'False').lower() == 'true',
}

# Create necessary directories
def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        STORAGE_SETTINGS['audio_dir'],
        STORAGE_SETTINGS['conversations_dir'],
        STORAGE_SETTINGS['reports_dir'],
        STORAGE_SETTINGS['models_dir'],
        STORAGE_SETTINGS['temp_dir'],
        os.path.dirname(LOGGING_SETTINGS['file']),
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

# Initialize directories
create_directories()


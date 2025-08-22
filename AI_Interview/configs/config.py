"""
Configuration file for AI Interview Evaluation System.
Centralizes all settings and parameters.
"""

import os

# Model Configuration
MODEL_CONFIG = {
    "model_id": "fixie-ai/ultravox-v0_5-llama-3_2-1b",
    "trust_remote_code": True,
    "max_new_tokens": 10000,
    "sampling_rate": 16000
}

# Audio Configuration
AUDIO_CONFIG = {
    "supported_formats": [".wav", ".mp3", ".flac", ".m4a"],
    "default_sr": 16000,
    "max_duration": 300  # 5 minutes max
}

# Paths Configuration
PATHS = {
    "audio_dir": "Audios",
    "output_dir": "outputs",
    "logs_dir": "logs"
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    "confidence_weight": 0.2,
    "pronunciation_weight": 0.2,
    "fluency_weight": 0.2,
    "emotional_tone_weight": 0.15,
    "grammar_weight": 0.15,
    "min_suitability_score": 0,
    "max_suitability_score": 100
}

# Create necessary directories
def create_directories():
    """Create necessary directories if they don't exist"""
    for path in [PATHS["output_dir"], PATHS["logs_dir"]]:
        os.makedirs(path, exist_ok=True)

# Initialize directories
create_directories() 
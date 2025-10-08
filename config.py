#!/usr/bin/env python3
"""
Configuration module for Ultravox API Server
"""

import os
from typing import List, Optional

class Config:
    """Configuration settings for the Ultravox API Server"""
    
    # Server configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    WORKERS: int = int(os.getenv("WORKERS", "1"))
    
    # Model configuration
    MODEL_NAME: str = os.getenv("MODEL_NAME", "fixie-ai/ultravox-v0_5-llama-3_2-1b")
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
    
    # External monitoring configuration
    EXTERNAL_LOGGING_ENABLED: bool = os.getenv("EXTERNAL_LOGGING_ENABLED", "false").lower() == "true"
    METRICS_ENDPOINT: Optional[str] = os.getenv("METRICS_ENDPOINT")
    LOG_FILE_PATH: str = os.getenv("LOG_FILE_PATH", "/var/log/ultravox/metrics.log")
    
    # Audio processing configuration
    AUDIO_CHUNK_SIZE: int = int(os.getenv("AUDIO_CHUNK_SIZE", "1048576"))  # 1MB chunks
    MAX_AUDIO_DURATION_SECONDS: int = int(os.getenv("MAX_AUDIO_DURATION_SECONDS", "300"))  # 5 minutes
    
    # Security configuration (configure properly for production)
    ALLOWED_HOSTS: List[str] = os.getenv("ALLOWED_HOSTS", "*").split(",")
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Logging configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration settings"""
        try:
            # Validate numeric values
            assert cls.PORT > 0 and cls.PORT < 65536, "Invalid port number"
            assert cls.WORKERS > 0, "Workers must be positive"
            assert cls.MAX_CONCURRENT_REQUESTS > 0, "Max concurrent requests must be positive"
            assert cls.AUDIO_CHUNK_SIZE > 0, "Audio chunk size must be positive"
            assert cls.MAX_AUDIO_DURATION_SECONDS > 0, "Max audio duration must be positive"
            
            return True
        except AssertionError as e:
            print(f"Configuration validation error: {e}")
            return False

# Global config instance
config = Config()


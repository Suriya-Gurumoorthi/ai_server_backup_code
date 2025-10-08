#!/usr/bin/env python3
"""
Audio Processing Module
Handles audio conversion, chunking, and streaming for memory efficiency
"""

import base64
import io
import numpy as np
import librosa
import soundfile as sf
import logging
from typing import Generator, Tuple, Optional, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Memory-efficient audio processing with chunked streaming support"""
    
    def __init__(self, chunk_size: int = 1024 * 1024):  # 1MB chunks by default
        self.chunk_size = chunk_size
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    async def base64_to_audio_async(self, audio_base64: str, target_sr: int = 16000) -> np.ndarray:
        """Convert base64 encoded audio to numpy array asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._base64_to_audio_sync,
            audio_base64,
            target_sr
        )
    
    def _base64_to_audio_sync(self, audio_base64: str, target_sr: int = 16000) -> np.ndarray:
        """Synchronous base64 to audio conversion"""
        try:
            # Decode base64
            audio_bytes = base64.b64decode(audio_base64)
            
            # Load audio using librosa
            audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
            
            # Resample to target sample rate if needed
            if sr != target_sr:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)
            
            return audio_array.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error converting base64 to audio: {e}")
            raise ValueError(f"Invalid audio data: {str(e)}")

    def base64_to_audio_chunked(self, audio_base64: str, target_sr: int = 16000) -> Generator[np.ndarray, None, None]:
        """Convert base64 audio in chunks for memory efficiency"""
        try:
            # Decode base64
            audio_bytes = base64.b64decode(audio_base64)
            
            # Load audio metadata first
            with io.BytesIO(audio_bytes) as audio_io:
                info = sf.info(audio_io)
                total_frames = info.frames
                sr = info.samplerate
                
                # Calculate chunk size in frames
                chunk_frames = int(self.chunk_size / (info.channels * info.subtype_info.get('bytes', 2)))
                
                # Process in chunks
                for start_frame in range(0, total_frames, chunk_frames):
                    end_frame = min(start_frame + chunk_frames, total_frames)
                    
                    # Load chunk
                    audio_io.seek(0)
                    chunk, _ = sf.read(audio_io, start=start_frame, stop=end_frame)
                    
                    # Resample if needed
                    if sr != target_sr:
                        chunk = librosa.resample(chunk, orig_sr=sr, target_sr=target_sr)
                    
                    yield chunk.astype(np.float32)
                    
        except Exception as e:
            logger.error(f"Error in chunked audio processing: {e}")
            raise ValueError(f"Invalid audio data: {str(e)}")

    async def process_audio_stream(self, audio_base64: str, target_sr: int = 16000) -> Tuple[np.ndarray, float]:
        """Process audio stream and return full audio array with duration"""
        loop = asyncio.get_event_loop()
        
        # For now, use the async method for full processing
        # In the future, this could be enhanced to process chunks as they arrive
        audio_array = await self.base64_to_audio_async(audio_base64, target_sr)
        duration = len(audio_array) / target_sr
        
        return audio_array, duration

    def validate_audio_data(self, audio_base64: str) -> bool:
        """Validate audio data without full processing"""
        try:
            # Quick validation - just decode base64
            audio_bytes = base64.b64decode(audio_base64)
            
            # Check if it's a valid audio format by trying to read metadata
            with io.BytesIO(audio_bytes) as audio_io:
                sf.info(audio_io)
            
            return True
        except Exception as e:
            logger.warning(f"Audio validation failed: {e}")
            return False

    def get_audio_info(self, audio_base64: str) -> dict:
        """Get audio information without full processing"""
        try:
            audio_bytes = base64.b64decode(audio_base64)
            
            with io.BytesIO(audio_bytes) as audio_io:
                info = sf.info(audio_io)
                
                return {
                    "duration_seconds": info.duration,
                    "sample_rate": info.samplerate,
                    "channels": info.channels,
                    "format": info.format,
                    "subtype": info.subtype,
                    "frames": info.frames,
                    "size_bytes": len(audio_bytes)
                }
        except Exception as e:
            logger.error(f"Error getting audio info: {e}")
            raise ValueError(f"Invalid audio data: {str(e)}")

    async def binary_to_audio_async(self, audio_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
        """Convert binary audio data to numpy array asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._binary_to_audio_sync,
            audio_bytes,
            target_sr
        )
    
    def _binary_to_audio_sync(self, audio_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
        """Synchronous binary to audio conversion"""
        try:
            # Load audio using librosa
            audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
            
            # Resample to target sample rate if needed
            if sr != target_sr:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)
            
            return audio_array.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error converting binary to audio: {e}")
            raise ValueError(f"Invalid audio data: {str(e)}")

    async def process_binary_audio_stream(self, audio_bytes: bytes, target_sr: int = 16000) -> Tuple[np.ndarray, float]:
        """Process binary audio stream and return full audio array with duration"""
        audio_array = await self.binary_to_audio_async(audio_bytes, target_sr)
        duration = len(audio_array) / target_sr
        
        return audio_array, duration

    def validate_binary_audio_data(self, audio_bytes: bytes) -> bool:
        """Validate binary audio data without full processing"""
        try:
            # Check if it's a valid audio format by trying to read metadata
            with io.BytesIO(audio_bytes) as audio_io:
                sf.info(audio_io)
            
            return True
        except Exception as e:
            logger.warning(f"Binary audio validation failed: {e}")
            return False

    def get_binary_audio_info(self, audio_bytes: bytes) -> dict:
        """Get audio information from binary data without full processing"""
        try:
            with io.BytesIO(audio_bytes) as audio_io:
                info = sf.info(audio_io)
                
                return {
                    "duration_seconds": info.duration,
                    "sample_rate": info.samplerate,
                    "channels": info.channels,
                    "format": info.format,
                    "subtype": info.subtype,
                    "frames": info.frames,
                    "size_bytes": len(audio_bytes)
                }
        except Exception as e:
            logger.error(f"Error getting binary audio info: {e}")
            raise ValueError(f"Invalid audio data: {str(e)}")

    def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)

# Global audio processor instance
audio_processor = AudioProcessor()


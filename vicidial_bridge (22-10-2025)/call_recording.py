"""
Call recording module for Vicidial Bridge

Handles complete call recording, separate audio tracks, and metadata generation.
"""

import logging
import os
import time
import wave
from datetime import datetime
from typing import Optional, Dict, Any

try:
    from .config import (
        SAVE_COMPLETE_CALLS, CALL_RECORDINGS_DIR, SAVE_SEPARATE_TRACKS, 
        SAVE_CALL_METADATA, VICIAL_SAMPLE_RATE, VAD_ENABLED, VAD_ENERGY_THRESHOLD,
        VAD_SILENCE_DURATION_MS, VAD_MIN_SPEECH_DURATION_MS, AUDIO_PROCESS_INTERVAL,
        LOCAL_MODEL_HOST, LOCAL_MODEL_PORT, TARGET_EXTENSION
    )
except ImportError:
    from config import (
        SAVE_COMPLETE_CALLS, CALL_RECORDINGS_DIR, SAVE_SEPARATE_TRACKS, 
        SAVE_CALL_METADATA, VICIAL_SAMPLE_RATE, VAD_ENABLED, VAD_ENERGY_THRESHOLD,
        VAD_SILENCE_DURATION_MS, VAD_MIN_SPEECH_DURATION_MS, AUDIO_PROCESS_INTERVAL,
        LOCAL_MODEL_HOST, LOCAL_MODEL_PORT, TARGET_EXTENSION
    )

logger = logging.getLogger(__name__)

class CallRecorder:
    """
    Records complete call conversations by capturing all audio from both directions
    Saves the entire call as a single WAV file for analysis
    """
    
    def __init__(self, call_id: str):
        self.call_id = call_id
        self.start_time = time.time()
        
        # Separate buffers for different audio sources
        self.vicial_audio_buffer = bytearray()  # Audio from caller (8kHz)
        self.ai_audio_buffer = bytearray()      # Audio from AI (8kHz)
        self.mixed_audio_buffer = bytearray()   # Mixed audio for complete recording
        
        # Recording statistics
        self.stats = {
            'vicial_audio_chunks': 0,
            'ai_audio_chunks': 0,
            'total_vicial_bytes': 0,
            'total_ai_bytes': 0,
            'recording_duration_ms': 0
        }
        
        logger.info(f"[{call_id}] CallRecorder initialized")
    
    def add_vicial_audio(self, audio_data: bytes):
        """Add audio from Vicial (caller) to recording buffer"""
        if not audio_data:
            return
        
        self.vicial_audio_buffer.extend(audio_data)
        self.mixed_audio_buffer.extend(audio_data)
        self.stats['vicial_audio_chunks'] += 1
        self.stats['total_vicial_bytes'] += len(audio_data)
        
        logger.debug(f"[{self.call_id}] Added Vicial audio: {len(audio_data)} bytes "
                    f"(total: {len(self.vicial_audio_buffer)} bytes)")
    
    def add_ai_audio(self, audio_data: bytes):
        """Add audio from AI to recording buffer"""
        if not audio_data:
            return
        
        self.ai_audio_buffer.extend(audio_data)
        self.mixed_audio_buffer.extend(audio_data)
        self.stats['ai_audio_chunks'] += 1
        self.stats['total_ai_bytes'] += len(audio_data)
        
        logger.debug(f"[{self.call_id}] Added AI audio: {len(audio_data)} bytes "
                    f"(total: {len(self.ai_audio_buffer)} bytes)")
    
    def get_recording_duration_ms(self) -> float:
        """Calculate total recording duration in milliseconds"""
        total_samples = len(self.mixed_audio_buffer) // 2  # 16-bit samples
        duration_ms = (total_samples / VICIAL_SAMPLE_RATE) * 1000
        self.stats['recording_duration_ms'] = duration_ms
        return duration_ms
    
    async def save_complete_call_recording(self) -> Optional[str]:
        """Save the complete call recording as a WAV file"""
        try:
            if not SAVE_COMPLETE_CALLS:
                logger.debug(f"[{self.call_id}] Call recording disabled")
                return None
            
            if not self.mixed_audio_buffer:
                logger.warning(f"[{self.call_id}] No audio data to save")
                return None
            
            # Create directory if it doesn't exist
            os.makedirs(CALL_RECORDINGS_DIR, exist_ok=True)
            
            # Create filename with timestamp and call info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_call_id = self.call_id.replace(":", "_").replace("/", "_")
            duration_sec = self.get_recording_duration_ms() / 1000
            
            # Create comprehensive filename
            filename = f"complete_call_{safe_call_id}_{timestamp}_{duration_sec:.1f}s.wav"
            filepath = os.path.join(CALL_RECORDINGS_DIR, filename)
            
            # Save complete call as WAV file
            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setnchannels(1)      # Mono
                wav_file.setsampwidth(2)      # 16-bit
                wav_file.setframerate(VICIAL_SAMPLE_RATE)  # 8kHz
                wav_file.writeframes(self.mixed_audio_buffer)
            
            # Create detailed metadata file if enabled
            if SAVE_CALL_METADATA:
                metadata_filename = filename.replace('.wav', '_metadata.txt')
                metadata_filepath = os.path.join(CALL_RECORDINGS_DIR, metadata_filename)
                
                with open(metadata_filepath, 'w', encoding='utf-8') as f:
                    f.write(f"Call Recording Metadata\n")
                    f.write(f"======================\n")
                    f.write(f"Call ID: {self.call_id}\n")
                    f.write(f"Start Time: {datetime.fromtimestamp(self.start_time).isoformat()}\n")
                    f.write(f"End Time: {datetime.now().isoformat()}\n")
                    f.write(f"Duration: {duration_sec:.2f} seconds\n")
                    f.write(f"Audio File: {filename}\n")
                    f.write(f"File Size: {len(self.mixed_audio_buffer)} bytes\n")
                    f.write(f"\nAudio Statistics:\n")
                    f.write(f"Vicial Audio Chunks: {self.stats['vicial_audio_chunks']}\n")
                    f.write(f"AI Audio Chunks: {self.stats['ai_audio_chunks']}\n")
                    f.write(f"Total Vicial Bytes: {self.stats['total_vicial_bytes']}\n")
                    f.write(f"Total AI Bytes: {self.stats['total_ai_bytes']}\n")
                    f.write(f"Mixed Audio Bytes: {len(self.mixed_audio_buffer)}\n")
                    f.write(f"Sample Rate: {VICIAL_SAMPLE_RATE} Hz\n")
                    f.write(f"Bit Depth: 16-bit\n")
                    f.write(f"Channels: Mono\n")
                    f.write(f"\nConfiguration:\n")
                    f.write(f"VAD Enabled: {VAD_ENABLED}\n")
                    f.write(f"VAD Energy Threshold: {VAD_ENERGY_THRESHOLD}\n")
                    f.write(f"VAD Silence Duration: {VAD_SILENCE_DURATION_MS}ms\n")
                    f.write(f"VAD Min Speech Duration: {VAD_MIN_SPEECH_DURATION_MS}ms\n")
                    f.write(f"Audio Process Interval: {AUDIO_PROCESS_INTERVAL*1000:.0f}ms\n")
                    f.write(f"Local Model Server: {LOCAL_MODEL_HOST}:{LOCAL_MODEL_PORT}\n")
                    f.write(f"Target Extension: {TARGET_EXTENSION}\n")
            
            # Log recording details
            logger.info(f"[{self.call_id}] 💾 COMPLETE CALL RECORDING SAVED")
            logger.info(f"[{self.call_id}] 📁 File: {filename}")
            logger.info(f"[{self.call_id}] 📊 Duration: {duration_sec:.2f}s")
            logger.info(f"[{self.call_id}] 📊 Size: {len(self.mixed_audio_buffer)} bytes")
            logger.info(f"[{self.call_id}] 📊 Vicial chunks: {self.stats['vicial_audio_chunks']}")
            logger.info(f"[{self.call_id}] 📊 AI chunks: {self.stats['ai_audio_chunks']}")
            logger.info(f"[{self.call_id}] 📁 Path: {filepath}")
            logger.info(f"[{self.call_id}] 📄 Metadata: {metadata_filepath}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"[{self.call_id}] Error saving complete call recording: {e}")
            return None
    
    async def save_separate_audio_tracks(self) -> Dict[str, str]:
        """Save separate audio tracks for caller and AI"""
        try:
            if not SAVE_COMPLETE_CALLS or not SAVE_SEPARATE_TRACKS:
                return {}
            
            saved_files = {}
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_call_id = self.call_id.replace(":", "_").replace("/", "_")
            
            # Save Vicial (caller) audio track
            if self.vicial_audio_buffer:
                vicial_filename = f"caller_audio_{safe_call_id}_{timestamp}.wav"
                vicial_filepath = os.path.join(CALL_RECORDINGS_DIR, vicial_filename)
                
                with wave.open(vicial_filepath, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(VICIAL_SAMPLE_RATE)
                    wav_file.writeframes(self.vicial_audio_buffer)
                
                saved_files['caller'] = vicial_filepath
                logger.info(f"[{self.call_id}] 💾 Caller audio saved: {vicial_filename}")
            
            # Save AI audio track
            if self.ai_audio_buffer:
                ai_filename = f"ai_audio_{safe_call_id}_{timestamp}.wav"
                ai_filepath = os.path.join(CALL_RECORDINGS_DIR, ai_filename)
                
                with wave.open(ai_filepath, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(VICIAL_SAMPLE_RATE)
                    wav_file.writeframes(self.ai_audio_buffer)
                
                saved_files['ai'] = ai_filepath
                logger.info(f"[{self.call_id}] 💾 AI audio saved: {ai_filename}")
            
            return saved_files
            
        except Exception as e:
            logger.error(f"[{self.call_id}] Error saving separate audio tracks: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get recording statistics"""
        return {
            'call_id': self.call_id,
            'start_time': self.start_time,
            'duration_ms': self.get_recording_duration_ms(),
            'vicial_audio_chunks': self.stats['vicial_audio_chunks'],
            'ai_audio_chunks': self.stats['ai_audio_chunks'],
            'total_vicial_bytes': self.stats['total_vicial_bytes'],
            'total_ai_bytes': self.stats['total_ai_bytes'],
            'mixed_audio_bytes': len(self.mixed_audio_buffer),
            'vicial_buffer_size': len(self.vicial_audio_buffer),
            'ai_buffer_size': len(self.ai_audio_buffer)
        }

"""
Text-to-Speech module for Vicidial Bridge

Handles high-quality TTS using Piper neural voices and audio format conversion.
"""

import logging
import os
import tempfile
import subprocess
import wave
import io
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

class PiperTTS:
    """High-quality TTS using Piper neural voices"""
    
    def __init__(self):
        self.voice_path = "/usr/piper_voices/en_US-lessac-medium.onnx"
        self.voice = None
        logger.info("PiperTTS initialized")
    
    async def initialize_voice(self):
        """Initialize the Piper voice model"""
        try:
            if not os.path.exists(self.voice_path):
                logger.error(f"Piper voice file not found: {self.voice_path}")
                return False
            
            import piper
            self.voice = piper.PiperVoice.load(self.voice_path)
            logger.info("✅ Piper voice model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Piper voice: {e}")
            return False
    
    async def generate_speech(self, text: str) -> Optional[bytes]:
        """Generate speech from text using Piper TTS"""
        try:
            if not self.voice:
                if not await self.initialize_voice():
                    return None
            
            # Create temporary file for audio output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_filename = temp_file.name
            
            try:
                # Use command-line piper for reliable synthesis
                cmd = [
                    'python3.11', '-m', 'piper',
                    '--model', self.voice_path,
                    '--output_file', temp_filename
                ]
                
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                stdout, stderr = process.communicate(input=text)
                
                if process.returncode != 0:
                    logger.error(f"Piper TTS failed: {stderr}")
                    return None
                
                # Read the generated WAV file
                with open(temp_filename, 'rb') as f:
                    wav_data = f.read()
                
                logger.info(f"🎵 Generated Piper TTS audio: {len(wav_data)} bytes")
                return wav_data
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)
            
        except Exception as e:
            logger.error(f"Error generating Piper TTS: {e}")
            return None
    
    def convert_wav_to_audiosocket_format(self, wav_data: bytes, target_sample_rate: int = 16000) -> Optional[bytes]:
        """Convert WAV audio to AudioSocket format (16kHz, 16-bit, mono)"""
        try:
            audio_stream = io.BytesIO(wav_data)
            with wave.open(audio_stream, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                
                logger.debug(f"📊 Original WAV: {frames} frames, {sample_rate}Hz, {channels} channels, {sample_width} bytes/sample")
                
                audio_data = wav_file.readframes(frames)
                
                if sample_width == 2:
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                elif sample_width == 4:
                    audio_array = np.frombuffer(audio_data, dtype=np.int32)
                else:
                    logger.error(f"❌ Unsupported sample width: {sample_width}")
                    return None
                
                # Convert stereo to mono if needed
                if channels == 2:
                    audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
                    logger.debug("🔄 Converted stereo to mono")
                elif channels > 2:
                    logger.error(f"❌ Unsupported channel count: {channels}")
                    return None
                
                # Resample if needed using proper decimation/upsampling
                if sample_rate != target_sample_rate:
                    if target_sample_rate < sample_rate:
                        # Downsampling: use decimation for exact 2x downsampling
                        if sample_rate == 16000 and target_sample_rate == 8000:
                            # Exact 2x decimation - take every 2nd sample
                            audio_array = audio_array[::2]
                            logger.debug(f"🔄 Decimated from {sample_rate}Hz to {target_sample_rate}Hz: {len(audio_array)} samples")
                        else:
                            # General downsampling with proper decimation
                            decimation_factor = sample_rate // target_sample_rate
                            audio_array = audio_array[::decimation_factor]
                            logger.debug(f"🔄 Decimated from {sample_rate}Hz to {target_sample_rate}Hz by factor {decimation_factor}: {len(audio_array)} samples")
                    else:
                        # Upsampling: use linear interpolation for upsampling
                        ratio = target_sample_rate / sample_rate
                        new_length = int(len(audio_array) * ratio)
                        old_indices = np.linspace(0, len(audio_array) - 1, new_length)
                        audio_array = np.interp(old_indices, np.arange(len(audio_array)), audio_array.astype(np.float32))
                        audio_array = audio_array.astype(np.int16)
                        logger.debug(f"🔄 Upsampled from {sample_rate}Hz to {target_sample_rate}Hz: {len(audio_array)} samples")
                
                audio_bytes = audio_array.tobytes()
                logger.debug(f"🎵 Converted to AudioSocket format: {len(audio_bytes)} bytes")
                return audio_bytes
                
        except Exception as e:
            logger.error(f"❌ Error converting audio format: {e}")
            return None

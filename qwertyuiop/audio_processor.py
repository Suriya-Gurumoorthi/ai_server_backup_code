"""
Audio processing module for TTS generation and audio handling.
Handles text-to-speech conversion and audio streaming.
"""

import io
import wave
import logging
import asyncio
from typing import Optional, Tuple

try:
    from piper import PiperVoice
except ImportError as e:
    print(f"Warning: Piper TTS not available: {e}")
    PiperVoice = None

from config import TTS_SAMPLE_RATE, MAX_TTS_TEXT_LENGTH
from utils import validate_text_for_tts, safe_send_response
from models import model_manager


class AudioProcessor:
    """Handles audio processing operations including TTS generation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def generate_tts_audio(self, text: str) -> Optional[bytes]:
        """Generate TTS audio from text using Piper."""
        if not model_manager.is_tts_available():
            self.logger.warning("TTS not available, cannot generate audio")
            return None
        
        # Validate and clean input text
        text = validate_text_for_tts(text)
        if not text:
            self.logger.warning("TTS text is empty or whitespace only. Skipping synthesis.")
            return None
        
        self.logger.info(f"TTS input text validation:")
        self.logger.info(f"  - Text type: {type(text)}")
        self.logger.info(f"  - Text length: {len(text) if text else 0}")
        self.logger.info(f"  - Text repr: {repr(text)}")
        
        try:
            self.logger.info(f"Calling PiperVoice.synthesize with text: {repr(text[:100])}...")
            
            # Generate audio using Piper TTS
            audio_data = model_manager.piper_voice.synthesize(text)
            self.logger.info(f"Piper synthesize returned type: {type(audio_data)}")
            
            # Process audio chunks
            audio_bytes = self._process_audio_chunks(audio_data)
            if not audio_bytes:
                return None
            
            # Convert to WAV format
            wav_bytes = self._convert_to_wav(audio_bytes)
            self.logger.info(f"Generated TTS audio: {len(wav_bytes)} bytes")
            return wav_bytes
            
        except Exception as e:
            self.logger.error(f"Error generating TTS audio: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def _process_audio_chunks(self, audio_data) -> Optional[bytes]:
        """Process audio chunks from Piper TTS output."""
        if hasattr(audio_data, '__iter__'):
            # Convert generator to list to examine the chunks
            audio_chunks = list(audio_data)
            self.logger.info(f"Piper synthesize returned {len(audio_chunks)} chunks")
            
            if len(audio_chunks) > 0:
                # Extract audio data from AudioChunk objects
                audio_bytes = b""
                for idx, chunk in enumerate(audio_chunks):
                    self.logger.info(f"Processing chunk {idx}:")
                    
                    # Check available attributes
                    if hasattr(chunk, 'audio_int16_bytes'):
                        chunk_bytes = chunk.audio_int16_bytes
                        self.logger.info(f"  - audio_int16_bytes: {len(chunk_bytes)} bytes")
                        if len(chunk_bytes) > 0:
                            audio_bytes += chunk_bytes
                            self.logger.info(f"  - Added {len(chunk_bytes)} bytes to total")
                        else:
                            self.logger.warning(f"  - audio_int16_bytes is empty!")
                    elif hasattr(chunk, 'audio_int16_array'):
                        chunk_array = chunk.audio_int16_array
                        chunk_bytes = chunk_array.tobytes()
                        self.logger.info(f"  - audio_int16_array: {chunk_array.shape}, {len(chunk_bytes)} bytes")
                        if len(chunk_bytes) > 0:
                            audio_bytes += chunk_bytes
                            self.logger.info(f"  - Added {len(chunk_bytes)} bytes to total")
                        else:
                            self.logger.warning(f"  - audio_int16_array is empty!")
                    else:
                        self.logger.warning(f"  - AudioChunk {idx} missing expected audio attributes")
                        self.logger.warning(f"  - Available attributes: {[attr for attr in dir(chunk) if not attr.startswith('_')]}")
                
                self.logger.info(f"Extracted audio bytes: {len(audio_bytes)} bytes")
                return audio_bytes
            else:
                self.logger.warning("No audio chunks received from Piper TTS")
                return None
        else:
            # If it's not a generator, treat as direct bytes
            return audio_data
    
    def _convert_to_wav(self, audio_bytes: bytes) -> bytes:
        """Convert raw audio bytes to WAV format."""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(TTS_SAMPLE_RATE)  # Piper default sample rate
            wav_file.writeframes(audio_bytes)
        
        return wav_buffer.getvalue()
    
    async def send_chunked_audio(self, websocket, audio_bytes: bytes, client_address: str, text_response: str) -> bool:
        """Send audio data in chunks to avoid WebSocket message size limits."""
        from config import CHUNK_SIZE
        
        try:
            # Send initial metadata
            metadata = {
                "type": "tts_start",
                "text": text_response,
                "audio_size": len(audio_bytes),
                "chunk_size": CHUNK_SIZE,
                "total_chunks": (len(audio_bytes) + CHUNK_SIZE - 1) // CHUNK_SIZE
            }
            
            self.logger.info(f"Sending TTS metadata to {client_address}: {len(audio_bytes)} bytes in {metadata['total_chunks']} chunks")
            await safe_send_response(websocket, str(metadata), client_address)
            
            # Send audio data in chunks
            chunk_count = 0
            for i in range(0, len(audio_bytes), CHUNK_SIZE):
                chunk = audio_bytes[i:i + CHUNK_SIZE]
                chunk_count += 1
                
                self.logger.info(f"Sending audio chunk {chunk_count}/{metadata['total_chunks']} ({len(chunk)} bytes) to {client_address}")
                await safe_send_response(websocket, chunk, client_address)
            
            # Send completion marker
            completion = {"type": "tts_end", "chunks_sent": chunk_count}
            self.logger.info(f"Sending TTS completion marker to {client_address}")
            await safe_send_response(websocket, str(completion), client_address)
            
            self.logger.info(f"Successfully sent {len(audio_bytes)} bytes of audio in {chunk_count} chunks to {client_address}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending chunked audio to {client_address}: {e}")
            return False


# Global audio processor instance
audio_processor = AudioProcessor()

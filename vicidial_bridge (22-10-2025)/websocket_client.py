"""
WebSocket client module for Vicidial Bridge

Handles communication with the local model server via WebSocket.
"""

import asyncio
import logging
import struct
import time
import wave
import io
from typing import Optional, Dict, Any

import websockets

try:
    from .config import LOCAL_MODEL_WS_URL, LOCAL_MODEL_SAMPLE_RATE
    from .audio_processing import AudioProcessor
    from .text_to_speech import PiperTTS
except ImportError:
    from config import LOCAL_MODEL_WS_URL, LOCAL_MODEL_SAMPLE_RATE
    from audio_processing import AudioProcessor
    from text_to_speech import PiperTTS

logger = logging.getLogger(__name__)

class LocalModelWebSocketClient:
    """
    WebSocket client for communicating with the local model server
    Handles audio processing via WebSocket binary messages
    """
    
    def __init__(self):
        self.ws_url = LOCAL_MODEL_WS_URL
        self.websocket = None
        self.call_id = None
        self.session_id = None
        self.audio_processor = AudioProcessor()
        self.conversation_history = []
        self.connected = False
        self.response_queue = asyncio.Queue()
        self.piper_tts = PiperTTS()  # Initialize Piper TTS
        
        logger.info(f"LocalModelWebSocketClient initialized - WebSocket: {self.ws_url}")
    
    def _pcm_to_wav_bytes(self, pcm_data: bytes, sample_rate: int, channels: int) -> bytes:
        """
        Convert raw PCM audio data to WAV format with proper headers
        
        Args:
            pcm_data: Raw 16-bit PCM audio data
            sample_rate: Sample rate in Hz (e.g., 16000)
            channels: Number of audio channels (1 for mono, 2 for stereo)
            
        Returns:
            WAV-formatted audio data as bytes
        """
        try:
            # Create a BytesIO buffer to write WAV data
            buf = io.BytesIO()
            
            # Use Python's wave module to create proper WAV headers
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(channels)      # Number of channels (1 = mono, 2 = stereo)
                wf.setsampwidth(2)             # Sample width in bytes (2 = 16-bit)
                wf.setframerate(sample_rate)   # Sample rate in Hz
                wf.writeframes(pcm_data)       # Write the raw PCM data
            
            # Get the complete WAV file data
            wav_bytes = buf.getvalue()
            buf.close()
            
            logger.debug(f"Converted {len(pcm_data)} bytes PCM to {len(wav_bytes)} bytes WAV")
            return wav_bytes
            
        except Exception as e:
            logger.error(f"Error converting PCM to WAV: {e}")
            # Fallback: return original PCM data if WAV conversion fails
            return pcm_data
    
    async def health_check(self) -> bool:
        """Check if the local model server is accessible via WebSocket"""
        try:
            logger.info(f"Checking local model WebSocket health at {self.ws_url}")
            
            # Try to connect to WebSocket with a short timeout
            try:
                async with asyncio.timeout(10):  # Use asyncio.timeout for Python 3.11+
                    async with websockets.connect(
                        self.ws_url,
                        ping_interval=None,  # Disable ping for health check
                        close_timeout=5
                    ) as test_ws:
                        logger.info("Local model WebSocket health check passed")
                        return True
            except asyncio.TimeoutError:
                logger.error("Local model WebSocket health check timed out")
                return False
            except Exception as e:
                logger.error(f"Local model WebSocket health check failed: {e}")
                return False
                        
        except Exception as e:
            logger.error(f"Local model WebSocket health check error: {e}")
            return False
    
    async def start_call(self) -> bool:
        """Start a new call session with local model server via WebSocket"""
        try:
            # Initialize session variables
            self.call_id = f"local_{int(time.time())}"
            self.session_id = f"session_{int(time.time())}"
            self.conversation_history = []
            
            # Connect to WebSocket
            logger.info(f"[LocalModel] Connecting to WebSocket: {self.ws_url}")
            self.websocket = await websockets.connect(
                self.ws_url,
                ping_interval=30,
                ping_timeout=300,
                close_timeout=10,
                max_size=10_000_000  # 10MB max message size
            )
            self.connected = True
            
            logger.info(f"[LocalModel] Started WebSocket call session: {self.call_id}")
            logger.info(f"[LocalModel] Ready for audio processing via WebSocket")
            
            # Start response listener task
            asyncio.create_task(self._listen_for_responses())
            
            return True
                
        except Exception as e:
            logger.error(f"Error starting local model WebSocket call: {e}")
            self.connected = False
            return False
    
    async def send_audio(self, audio_data: bytes):
        """Send PCM audio data to local model via WebSocket"""
        if not audio_data or not self.connected or not self.websocket:
            logger.warning(f"[LocalModel] Cannot send audio: data={len(audio_data) if audio_data else 0}, connected={self.connected}, ws={self.websocket is not None}")
            return
        
        try:
            # Log audio data characteristics for debugging
            samples = len(audio_data) // 2
            duration_ms = (samples / LOCAL_MODEL_SAMPLE_RATE) * 1000
            max_amplitude = max(abs(s) for s in struct.unpack(f'<{samples}h', audio_data)) if samples > 0 else 0
            
            logger.info(f"[LocalModel] Sending audio: {len(audio_data)} bytes, {samples} samples, {duration_ms:.1f}ms, max_amp={max_amplitude}")
            
            # Convert raw PCM to WAV format for the Ultravox server
            wav_bytes = self._pcm_to_wav_bytes(audio_data, LOCAL_MODEL_SAMPLE_RATE, 1)
            logger.info(f"[LocalModel] Converted to WAV: {len(wav_bytes)} bytes")
            
            # Send audio as binary message to WebSocket (Ultravox expects raw audio, not JSON + binary)
            await self.websocket.send(wav_bytes)
            logger.info(f"[LocalModel] ✅ Sent {len(wav_bytes)} bytes of audio via WebSocket")
            
        except Exception as e:
            logger.error(f"Error sending audio to local model WebSocket: {e}")
            self.connected = False
    
    async def _listen_for_responses(self):
        """Listen for responses from the WebSocket server - now handles direct audio responses"""
        try:
            while self.connected and self.websocket:
                try:
                    # Wait for response from WebSocket
                    response = await self.websocket.recv()
                    
                    if isinstance(response, str):
                        # Text response from the AI (legacy support)
                        logger.info(f"[LocalModel] 📝 Received text response: {response[:100]}...")
                        
                        # Add user turn to conversation history
                        self.conversation_history.append({
                            "role": "user",
                            "content": "[Audio input]"
                        })
                        
                        # Add AI response to conversation history
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": str(response)
                        })
                        
                        # Put response in queue for processing
                        await self.response_queue.put({
                            "type": "text",
                            "content": response,
                            "timestamp": time.time()
                        })
                        
                    elif isinstance(response, bytes):
                        # Binary response - now primarily audio from Ultravox
                        logger.info(f"[LocalModel] 🎵 Received binary response: {len(response)} bytes")
                        
                        # Check if it's WAV audio data (most common case now)
                        if response.startswith(b'RIFF') and b'WAVE' in response[:12]:
                            logger.info(f"[LocalModel] 🎵 Detected WAV audio response from Ultravox")
                            await self.response_queue.put({
                                "type": "audio",
                                "content": response,
                                "timestamp": time.time()
                            })
                        else:
                            # Handle raw PCM audio data (alternative format)
                            logger.info(f"[LocalModel] 🎵 Processing as raw PCM audio data")
                            await self.response_queue.put({
                                "type": "audio",
                                "content": response,
                                "timestamp": time.time()
                            })
                    
                except websockets.ConnectionClosed:
                    logger.info("[LocalModel] WebSocket connection closed")
                    self.connected = False
                    break
                except Exception as e:
                    logger.error(f"[LocalModel] Error receiving WebSocket response: {e}")
                    self.connected = False
                    break
                    
        except Exception as e:
            logger.error(f"[LocalModel] Error in WebSocket response listener: {e}")
            self.connected = False
    
    async def get_response(self, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Get the next response from the WebSocket server"""
        try:
            if not self.connected:
                return None
                
            # Wait for response with timeout
            response = await asyncio.wait_for(self.response_queue.get(), timeout=timeout)
            return response
            
        except asyncio.TimeoutError:
            logger.debug("[LocalModel] Timeout waiting for WebSocket response")
            return None
        except Exception as e:
            logger.error(f"[LocalModel] Error getting WebSocket response: {e}")
            return None
    
    async def close_call(self):
        """Close the local model WebSocket call session"""
        try:
            self.connected = False
            
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
            
            # Clear response queue
            while not self.response_queue.empty():
                try:
                    self.response_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            # Clean up session variables
            self.call_id = None
            self.session_id = None
            self.conversation_history = []
            
            logger.info("[LocalModel] WebSocket call session cleaned up")
                
        except Exception as e:
            logger.error(f"Error closing local model WebSocket call: {e}")

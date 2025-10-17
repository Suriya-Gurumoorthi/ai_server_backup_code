"""
Additional methods for VicialLocalModelBridge

Contains helper methods and response processing logic for the bridge.
"""

import asyncio
import logging
import struct
import time
import wave
import base64
import numpy as np
from array import array
from datetime import datetime
from typing import Optional, Dict, Any

try:
    from .config import VICIAL_SAMPLE_RATE, LOCAL_MODEL_SAMPLE_RATE, SAVE_AI_AUDIO, AI_AUDIO_DIR, AUDIO_CHUNK_SIZE
    from .audio_processing import AudioSocketProtocol
except ImportError:
    from config import VICIAL_SAMPLE_RATE, LOCAL_MODEL_SAMPLE_RATE, SAVE_AI_AUDIO, AI_AUDIO_DIR, AUDIO_CHUNK_SIZE
    from audio_processing import AudioSocketProtocol

logger = logging.getLogger(__name__)

class BridgeMethods:
    """Helper methods for VicialLocalModelBridge"""
    
    def __init__(self, bridge_instance):
        self.bridge = bridge_instance
    
    async def _stream_audio_chunk(self, call_id: str, audio_chunk: bytes):
        """Stream a chunk of audio to Local Model"""
        # IMMEDIATE EXIT: Check if call is still active before processing
        if call_id not in self.bridge.active_calls:
            logger.debug(f"[{call_id}] Skipping audio chunk - call no longer active")
            return
            
        call_info = self.bridge.active_calls.get(call_id)
        if not call_info or call_info.get('call_ended', False):
            logger.debug(f"[{call_id}] Skipping audio chunk - call ended")
            return
        
        try:
            local_model_client = call_info['local_model_client']
            
            # Send audio to Local Model
            await local_model_client.send_audio(audio_chunk)
            logger.debug(f"[{call_id}] 📤 Sent {len(audio_chunk)} bytes of audio to Local Model")
            
        except Exception as e:
            logger.error(f"[{call_id}] Error streaming audio chunk: {e}")
        finally:
            # Mark processing as complete
            if call_info:
                call_info['processing_audio'] = False
    
    async def _process_local_model_responses(self, call_id: str):
        """Process responses from Local Model WebSocket"""
        call_info = self.bridge.active_calls.get(call_id)
        if not call_info:
            return
        
        local_model_client = call_info['local_model_client']
        
        logger.info(f"[{call_id}] Starting Local Model WebSocket response processing")
        
        try:
            while call_id in self.bridge.active_calls:
                # IMMEDIATE EXIT: Check if call is still active at the start of each iteration
                if call_id not in self.bridge.active_calls:
                    logger.info(f"[{call_id}] Local Model response processing stopped - call no longer active")
                    break
                
                # Check if call has ended
                call_info = self.bridge.active_calls.get(call_id)
                if call_info and call_info.get('call_ended', False):
                    logger.info(f"[{call_id}] Local Model response processing stopped - call ended")
                    break
                
                # Get response from WebSocket with short timeout
                try:
                    response = await asyncio.wait_for(local_model_client.get_response(timeout=1.0), timeout=1.5)
                except asyncio.TimeoutError:
                    # Timeout is normal, continue loop
                    continue
                except asyncio.CancelledError:
                    logger.info(f"[{call_id}] Local Model response processing cancelled")
                    break
                
                if response:
                    if response["type"] == "text":
                        # Process text response from AI - now server handles TTS
                        ai_text = response["content"]
                        call_info['ai_responses_received'] += 1
                        logger.info(f"[{call_id}] 📝 Received AI text response #{call_info['ai_responses_received']}: {ai_text[:100]}...")
                        
                        # Server will handle TTS and send audio, so we just log the text
                        logger.info(f"[{call_id}] 📝 Text response logged, waiting for server TTS audio...")
                        
                    elif response["type"] == "audio":
                        # Process audio response from server TTS
                        audio_data = response["content"]
                        call_info['ai_responses_received'] += 1
                        logger.info(f"[{call_id}] 🎵 Received server TTS audio response #{call_info['ai_responses_received']}: {len(audio_data)} bytes")
                        
                        # Process server TTS audio response with current turn ID
                        current_turn_id = call_info.get('current_turn_id', 0)
                        await self._process_server_tts_audio_response(call_id, audio_data, current_turn_id)
                        
                    elif response["type"] == "binary":
                        # Process binary response (could be audio)
                        binary_data = response["content"]
                        logger.info(f"[{call_id}] 📦 Received binary response: {len(binary_data)} bytes")
                        
                        # Try to process as audio if it looks like WAV data
                        if binary_data.startswith(b'RIFF') and b'WAVE' in binary_data[:12]:
                            logger.info(f"[{call_id}] 🎵 Processing binary data as WAV audio")
                            await self._process_server_tts_audio_response(call_id, binary_data, call_info.get('current_turn_id', 0))
                        else:
                            # Assume it's raw audio from server TTS
                            logger.info(f"[{call_id}] 🎵 Processing binary data as server TTS audio")
                            await self._process_server_tts_audio_response(call_id, binary_data, call_info.get('current_turn_id', 0))
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"[{call_id}] Error in Local Model WebSocket processing: {e}")
            # Reset audio streaming flags on error to prevent getting stuck
            if call_info:
                call_info['audio_sending_in_progress'] = False
                call_info['audio_streaming_complete'] = True
                call_info['audio_streaming_failed'] = True
    
    # Note: _process_ai_text_response method removed - server now handles TTS directly
    
    async def _process_server_tts_audio_response(self, call_id: str, audio_data: bytes, turn_id: int):
        """Process server TTS audio response and send back to Vicial with turn control"""
        # IMMEDIATE EXIT: Check if call is still active before processing
        if call_id not in self.bridge.active_calls:
            logger.debug(f"[{call_id}] Skipping server TTS audio processing - call no longer active")
            return
            
        call_info = self.bridge.active_calls.get(call_id)
        if not call_info or call_info.get('call_ended', False):
            logger.debug(f"[{call_id}] Skipping server TTS audio processing - call ended")
            return
        
        try:
            logger.info(f"[{call_id}] 🎵 Processing server TTS audio response: {len(audio_data)} bytes")
            
            # Check if it's WAV format or raw audio
            if audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:12]:
                # It's WAV format, convert to PCM
                logger.info(f"[{call_id}] 🎵 Detected WAV format, converting to PCM")
                pcm_16k = call_info['local_model_client'].piper_tts.convert_wav_to_audiosocket_format(audio_data, target_sample_rate=16000)
            else:
                # Assume it's already PCM audio data
                logger.info(f"[{call_id}] 🎵 Assuming raw PCM audio data")
                pcm_16k = audio_data
            
            if pcm_16k:
                logger.info(f"[{call_id}] 🎵 Server TTS audio ready: {len(pcm_16k)} bytes")
                
                # Downsample from 16kHz to 8kHz for Vicial
                pcm_8k = self.bridge.audio_processor.downsample_16k_to_8k(pcm_16k)
                
                if pcm_8k:
                    # COMPLETE CALL RECORDING: Add server TTS audio to call recorder
                    call_recorder = call_info.get('call_recorder')
                    if call_recorder:
                        call_recorder.add_ai_audio(pcm_8k)
                    
                    # BARGE-IN SUPPORT: Check if this turn is still current before sending
                    if call_info.get('current_turn_id', 0) != turn_id:
                        logger.info(f"[{call_id}] 🚫 Server TTS audio playback cancelled - turn {turn_id} is no longer current (current: {call_info.get('current_turn_id', 0)})")
                        return
                    
                    # CRITICAL FIX: Prevent race conditions - only allow one audio send at a time
                    if call_info.get('audio_sending_in_progress', False):
                        logger.warning(f"[{call_id}] Audio send already in progress - skipping server TTS audio to prevent race condition")
                        return
                    
                    call_info['audio_sending_in_progress'] = True
                    call_info['audio_streaming_complete'] = False  # Reset completion flag
                    call_info['audio_streaming_failed'] = False    # Reset failure flag
                    
                    try:
                        # BARGE-IN SUPPORT: Create cancellable server TTS audio playback task
                        tts_task = asyncio.create_task(self._send_server_tts_audio_with_turn_control(call_id, pcm_8k, turn_id))
                        call_info['tts_playback_task'] = tts_task
                        
                        # Wait for audio playback to complete or be cancelled
                        try:
                            success = await tts_task
                        except asyncio.CancelledError:
                            logger.info(f"[{call_id}] 🚫 Server TTS audio playback task was cancelled (barge-in detected)")
                            success = False
                        finally:
                            call_info['tts_playback_task'] = None
                        
                        if success:
                            logger.info(f"[{call_id}] 🔊 Sent server TTS audio to Vicial: {len(pcm_8k)} bytes (real-time chunked)")
                            call_info['audio_frames_received'] += 1
                            
                            # CRITICAL FIX: Mark audio streaming as complete
                            call_info['audio_streaming_complete'] = True
                            
                            # Save audio for debugging if enabled
                            if SAVE_AI_AUDIO:
                                await self._save_server_tts_audio_response(call_id, audio_data, pcm_8k)
                        else:
                            logger.error(f"[{call_id}] Failed to send server TTS audio to Vicial")
                            # Mark as failed so cleanup knows
                            call_info['audio_streaming_failed'] = True
                    except Exception as e:
                        logger.error(f"[{call_id}] Exception during server TTS audio playback: {e}")
                        call_info['audio_streaming_failed'] = True
                    finally:
                        # Always clear the in-progress flag
                        call_info['audio_sending_in_progress'] = False
                else:
                    logger.error(f"[{call_id}] Failed to downsample server TTS audio from 16kHz to 8kHz")
            else:
                logger.error(f"[{call_id}] Failed to process server TTS audio data")
                        
        except Exception as e:
            logger.error(f"[{call_id}] Error processing server TTS audio response: {e}")

    async def _process_ai_audio_response(self, call_id: str, audio_wav_data: bytes, turn_id: int):
        """Process direct audio response from Ultravox and send back to Vicial with turn control"""
        # IMMEDIATE EXIT: Check if call is still active before processing
        if call_id not in self.bridge.active_calls:
            logger.debug(f"[{call_id}] Skipping AI audio processing - call no longer active")
            return
            
        call_info = self.bridge.active_calls.get(call_id)
        if not call_info or call_info.get('call_ended', False):
            logger.debug(f"[{call_id}] Skipping AI audio processing - call ended")
            return
        
        try:
            logger.info(f"[{call_id}] 🎵 Processing direct audio response: {len(audio_wav_data)} bytes")
            
            # Convert WAV to 16kHz PCM using Piper TTS converter
            pcm_16k = call_info['local_model_client'].piper_tts.convert_wav_to_audiosocket_format(audio_wav_data, target_sample_rate=16000)
            
            if pcm_16k:
                logger.info(f"[{call_id}] 🎵 Converted Ultravox audio to 16kHz PCM: {len(pcm_16k)} bytes")
                
                # Downsample from 16kHz to 8kHz for Vicial
                pcm_8k = self.bridge.audio_processor.downsample_16k_to_8k(pcm_16k)
                
                if pcm_8k:
                    # COMPLETE CALL RECORDING: Add Ultravox AI audio to call recorder
                    call_recorder = call_info.get('call_recorder')
                    if call_recorder:
                        call_recorder.add_ai_audio(pcm_8k)
                    
                    # BARGE-IN SUPPORT: Check if this turn is still current before sending
                    if call_info.get('current_turn_id', 0) != turn_id:
                        logger.info(f"[{call_id}] 🚫 Ultravox audio playback cancelled - turn {turn_id} is no longer current (current: {call_info.get('current_turn_id', 0)})")
                        return
                    
                    # CRITICAL FIX: Prevent race conditions - only allow one audio send at a time
                    if call_info.get('audio_sending_in_progress', False):
                        logger.warning(f"[{call_id}] Audio send already in progress - skipping Ultravox audio to prevent race condition")
                        return
                    
                    call_info['audio_sending_in_progress'] = True
                    call_info['audio_streaming_complete'] = False  # Reset completion flag
                    call_info['audio_streaming_failed'] = False    # Reset failure flag
                    
                    try:
                        # BARGE-IN SUPPORT: Create cancellable Ultravox audio playback task
                        ultravox_task = asyncio.create_task(self._send_ultravox_audio_with_turn_control(call_id, pcm_8k, turn_id))
                        call_info['tts_playback_task'] = ultravox_task
                        
                        # Wait for audio playback to complete or be cancelled
                        try:
                            success = await ultravox_task
                        except asyncio.CancelledError:
                            logger.info(f"[{call_id}] 🚫 Ultravox audio playback task was cancelled (barge-in detected)")
                            success = False
                        finally:
                            call_info['tts_playback_task'] = None
                        
                        if success:
                            logger.info(f"[{call_id}] 🔊 Sent Ultravox audio to Vicial: {len(pcm_8k)} bytes (real-time chunked)")
                            call_info['audio_frames_received'] += 1
                            
                            # CRITICAL FIX: Mark audio streaming as complete
                            call_info['audio_streaming_complete'] = True
                            
                            # Save audio for debugging if enabled
                            if SAVE_AI_AUDIO:
                                await self._save_ultravox_audio_response(call_id, audio_wav_data, pcm_8k)
                        else:
                            logger.error(f"[{call_id}] Failed to send Ultravox audio to Vicial")
                            # Mark as failed so cleanup knows
                            call_info['audio_streaming_failed'] = True
                    except Exception as e:
                        logger.error(f"[{call_id}] Exception during Ultravox audio playback: {e}")
                        call_info['audio_streaming_failed'] = True
                    finally:
                        # Always clear the in-progress flag
                        call_info['audio_sending_in_progress'] = False
                else:
                    logger.error(f"[{call_id}] Failed to downsample Ultravox audio from 16kHz to 8kHz")
            else:
                logger.error(f"[{call_id}] Failed to convert Ultravox WAV to PCM")
                        
        except Exception as e:
            logger.error(f"[{call_id}] Error processing Ultravox audio response: {e}")
    
    # Note: _send_tts_audio_to_vicial method removed - server now handles TTS directly
    
    async def _send_tts_audio_with_turn_control(self, call_id: str, pcm_8k: bytes, turn_id: int):
        """Send TTS audio with turn-based validation to support barge-in"""
        try:
            call_info = self.bridge.active_calls.get(call_id)
            if not call_info:
                return False
            
            writer = call_info['vicial_writer']
            
            # Send audio with turn validation
            success = await AudioSocketProtocol.write_audio_chunked_with_turn_control(
                writer, pcm_8k, chunk_size=AUDIO_CHUNK_SIZE, sample_rate=8000, 
                turn_validator=lambda: call_info.get('current_turn_id', 0) == turn_id
            )
            
            return success
            
        except asyncio.CancelledError:
            logger.info(f"[{call_id}] TTS audio sending cancelled for turn {turn_id}")
            raise
        except Exception as e:
            logger.error(f"[{call_id}] Error in turn-controlled TTS sending: {e}")
            return False
    
    async def _send_server_tts_audio_with_turn_control(self, call_id: str, pcm_8k: bytes, turn_id: int):
        """Send server TTS audio with turn-based validation to support barge-in"""
        try:
            call_info = self.bridge.active_calls.get(call_id)
            if not call_info:
                return False
            
            writer = call_info['vicial_writer']
            
            # Send audio with turn validation
            success = await AudioSocketProtocol.write_audio_chunked_with_turn_control(
                writer, pcm_8k, chunk_size=AUDIO_CHUNK_SIZE, sample_rate=8000, 
                turn_validator=lambda: call_info.get('current_turn_id', 0) == turn_id
            )
            
            return success
            
        except asyncio.CancelledError:
            logger.info(f"[{call_id}] Server TTS audio sending cancelled for turn {turn_id}")
            raise
        except Exception as e:
            logger.error(f"[{call_id}] Error in turn-controlled server TTS sending: {e}")
            return False

    async def _send_ultravox_audio_with_turn_control(self, call_id: str, pcm_8k: bytes, turn_id: int):
        """Send Ultravox audio with turn-based validation to support barge-in"""
        try:
            call_info = self.bridge.active_calls.get(call_id)
            if not call_info:
                return False
            
            writer = call_info['vicial_writer']
            
            # Send audio with turn validation
            success = await AudioSocketProtocol.write_audio_chunked_with_turn_control(
                writer, pcm_8k, chunk_size=AUDIO_CHUNK_SIZE, sample_rate=8000, 
                turn_validator=lambda: call_info.get('current_turn_id', 0) == turn_id
            )
            
            return success
            
        except asyncio.CancelledError:
            logger.info(f"[{call_id}] Ultravox audio sending cancelled for turn {turn_id}")
            raise
        except Exception as e:
            logger.error(f"[{call_id}] Error in turn-controlled Ultravox sending: {e}")
            return False
    
    async def _save_server_tts_audio_response(self, call_id: str, server_audio_data: bytes, pcm_8k_data: bytes):
        """Save server TTS audio response to file for debugging"""
        try:
            if not SAVE_AI_AUDIO:
                return
            
            # Create directory if it doesn't exist
            import os
            os.makedirs(AI_AUDIO_DIR, exist_ok=True)
            
            # Create filename with timestamp and call info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_call_id = call_id.replace(":", "_").replace("/", "_")
            
            # Save original server audio file
            audio_filename = f"server_tts_response_{safe_call_id}_{timestamp}.wav"
            audio_filepath = os.path.join(AI_AUDIO_DIR, audio_filename)
            
            with open(audio_filepath, 'wb') as f:
                f.write(server_audio_data)
            
            # Save converted 8kHz PCM file
            pcm_filename = f"server_tts_response_8k_{safe_call_id}_{timestamp}.wav"
            pcm_filepath = os.path.join(AI_AUDIO_DIR, pcm_filename)
            
            # Create WAV file with proper headers for 8kHz PCM
            with wave.open(pcm_filepath, 'wb') as wav_file:
                wav_file.setnchannels(1)      # Mono
                wav_file.setsampwidth(2)      # 16-bit
                wav_file.setframerate(8000)   # 8kHz
                wav_file.writeframes(pcm_8k_data)
            
            logger.info(f"[{call_id}] 💾 Saved server TTS audio files:")
            logger.info(f"[{call_id}]   - Original server audio: {audio_filepath}")
            logger.info(f"[{call_id}]   - 8kHz PCM: {pcm_filepath}")
            
        except Exception as e:
            logger.error(f"[{call_id}] Error saving server TTS audio response: {e}")

    async def _save_tts_audio_response(self, call_id: str, tts_wav_data: bytes, pcm_8k_data: bytes):
        """Save TTS audio response to file for debugging"""
        try:
            if not SAVE_AI_AUDIO:
                return
            
            # Create directory if it doesn't exist
            import os
            os.makedirs(AI_AUDIO_DIR, exist_ok=True)
            
            # Create filename with timestamp and call info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_call_id = call_id.replace(":", "_").replace("/", "_")
            
            # Save original WAV file
            wav_filename = f"tts_response_{safe_call_id}_{timestamp}.wav"
            wav_filepath = os.path.join(AI_AUDIO_DIR, wav_filename)
            
            with open(wav_filepath, 'wb') as f:
                f.write(tts_wav_data)
            
            # Save converted 8kHz PCM file
            pcm_filename = f"tts_response_8k_{safe_call_id}_{timestamp}.wav"
            pcm_filepath = os.path.join(AI_AUDIO_DIR, pcm_filename)
            
            # Create WAV file with proper headers for 8kHz PCM
            with wave.open(pcm_filepath, 'wb') as wav_file:
                wav_file.setnchannels(1)      # Mono
                wav_file.setsampwidth(2)      # 16-bit
                wav_file.setframerate(8000)   # 8kHz
                wav_file.writeframes(pcm_8k_data)
            
            logger.info(f"[{call_id}] 💾 Saved TTS audio files:")
            logger.info(f"[{call_id}]   - Original WAV: {wav_filepath}")
            logger.info(f"[{call_id}]   - 8kHz PCM: {pcm_filepath}")
            
        except Exception as e:
            logger.error(f"[{call_id}] Error saving TTS audio response: {e}")
    
    async def _save_ultravox_audio_response(self, call_id: str, ultravox_wav_data: bytes, pcm_8k_data: bytes):
        """Save Ultravox audio response to file for debugging"""
        try:
            if not SAVE_AI_AUDIO:
                return
            
            # Create directory if it doesn't exist
            import os
            os.makedirs(AI_AUDIO_DIR, exist_ok=True)
            
            # Create filename with timestamp and call info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_call_id = call_id.replace(":", "_").replace("/", "_")
            
            # Save original Ultravox WAV file
            wav_filename = f"ultravox_response_{safe_call_id}_{timestamp}.wav"
            wav_filepath = os.path.join(AI_AUDIO_DIR, wav_filename)
            
            with open(wav_filepath, 'wb') as f:
                f.write(ultravox_wav_data)
            
            # Save converted 8kHz PCM file
            pcm_filename = f"ultravox_response_8k_{safe_call_id}_{timestamp}.wav"
            pcm_filepath = os.path.join(AI_AUDIO_DIR, pcm_filename)
            
            # Create WAV file with proper headers for 8kHz PCM
            with wave.open(pcm_filepath, 'wb') as wav_file:
                wav_file.setnchannels(1)      # Mono
                wav_file.setsampwidth(2)      # 16-bit
                wav_file.setframerate(8000)   # 8kHz
                wav_file.writeframes(pcm_8k_data)
            
            logger.info(f"[{call_id}] 💾 Saved Ultravox audio files:")
            logger.info(f"[{call_id}]   - Original WAV: {wav_filepath}")
            logger.info(f"[{call_id}]   - 8kHz PCM: {pcm_filepath}")
            
        except Exception as e:
            logger.error(f"[{call_id}] Error saving Ultravox audio response: {e}")
    
    async def _inject_test_audio(self, call_id: str, writer):
        """Inject test audio to verify AudioSocket is working"""
        try:
            logger.info(f"[{call_id}] 🎵 Injecting test audio to verify AudioSocket...")
            
            # Generate test tone sequence (musical chord)
            tones = [440, 554, 659, 880]  # A, C#, E, A
            duration_per_tone = 0.5
            total_audio = bytearray()
            
            for freq in tones:
                samples = int(VICIAL_SAMPLE_RATE * duration_per_tone)
                tone_data = array("h")
                
                for i in range(samples):
                    sample = int(1000 * np.sin(2 * np.pi * freq * i / VICIAL_SAMPLE_RATE))
                    tone_data.append(sample)
                
                total_audio.extend(tone_data.tobytes())
            
            test_audio = bytes(total_audio)
            
            # Analyze test audio
            samples = len(test_audio) // 2
            duration_ms = (samples / VICIAL_SAMPLE_RATE) * 1000
            max_amp = max(abs(s) for s in struct.unpack(f'<{samples}h', test_audio)) if samples > 0 else 0
            logger.info(f"[{call_id}] 📊 Test audio: {len(test_audio)} bytes, {samples} samples, {duration_ms:.1f}ms, max_amp={max_amp}")
            
            # Send test audio
            success = await AudioSocketProtocol.write_audio_frame(writer, test_audio)
            
            if success:
                logger.info(f"[{call_id}] ✅ Test audio injected successfully")
                logger.info(f"[{call_id}] 🔊 You should hear a musical chord sequence in the call")
                return True
            else:
                logger.error(f"[{call_id}] ❌ Failed to inject test audio")
                return False
                
        except Exception as e:
            logger.error(f"[{call_id}] Error injecting test audio: {e}")
            return False
    
    async def _inject_mid_call_test_tone(self, call_id: str, writer, tone_type: str = "beep"):
        """Inject a test tone mid-call to verify AudioSocket is still active"""
        try:
            call_info = self.bridge.active_calls.get(call_id)
            if not call_info:
                return False
            
            call_info['test_tones_sent'] += 1
            call_info['last_test_tone_time'] = time.time()
            
            logger.info(f"[{call_id}] 🎵 Injecting mid-call test tone #{call_info['test_tones_sent']} ({tone_type})...")
            
            if tone_type == "beep":
                # Generate a simple beep (800Hz for 0.5 seconds)
                frequency = 800
                duration = 0.5
            elif tone_type == "chord":
                # Generate a musical chord
                frequency = 440
                duration = 1.0
            else:
                # Default to 440Hz
                frequency = 440
                duration = 0.5
            
            samples = int(VICIAL_SAMPLE_RATE * duration)
            tone_data = array("h")
            
            for i in range(samples):
                sample = int(1000 * np.sin(2 * np.pi * frequency * i / VICIAL_SAMPLE_RATE))
                tone_data.append(sample)
            
            test_audio = tone_data.tobytes()
            
            # Analyze test audio
            samples_count = len(test_audio) // 2
            duration_ms = (samples_count / VICIAL_SAMPLE_RATE) * 1000
            max_amp = max(abs(s) for s in struct.unpack(f'<{samples_count}h', test_audio)) if samples_count > 0 else 0
            logger.info(f"[{call_id}] 📊 Mid-call tone: {len(test_audio)} bytes, {samples_count} samples, {duration_ms:.1f}ms, max_amp={max_amp}")
            
            # Send test audio
            success = await AudioSocketProtocol.write_audio_frame(writer, test_audio)
            
            if success:
                logger.info(f"[{call_id}] ✅ Mid-call test tone #{call_info['test_tones_sent']} sent successfully")
                logger.info(f"[{call_id}] 🔊 You should hear a {frequency}Hz tone in the call")
                return True
            else:
                logger.error(f"[{call_id}] ❌ Failed to send mid-call test tone")
                return False
                
        except Exception as e:
            logger.error(f"[{call_id}] Error injecting mid-call test tone: {e}")
            return False
    
    async def _send_error_tone(self, writer):
        """Send an error tone to indicate call failure"""
        try:
            # Generate a simple error tone (440Hz for 1 second at 8kHz)
            duration = 1.0  # 1 second
            samples = int(VICIAL_SAMPLE_RATE * duration)
            tone_data = array("h")
            
            for i in range(samples):
                # Generate 440Hz sine wave
                sample = int(1000 * np.sin(2 * np.pi * 440 * i / VICIAL_SAMPLE_RATE))
                tone_data.append(sample)
            
            await AudioSocketProtocol.write_audio_frame(writer, tone_data.tobytes())
            logger.info("Sent error tone to caller")
            
        except Exception as e:
            logger.error(f"Error sending error tone: {e}")
    
    async def _monitor_call_health(self, call_id: str):
        """Monitor call health and handle timeouts"""
        try:
            while call_id in self.bridge.active_calls:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # IMMEDIATE EXIT: Check if call is still active
                if call_id not in self.bridge.active_calls:
                    logger.info(f"[{call_id}] Call health monitoring stopped - call no longer active")
                    break
                
                # Check if call has ended
                call_info = self.bridge.active_calls.get(call_id)
                if call_info and call_info.get('call_ended', False):
                    logger.info(f"[{call_id}] Call health monitoring stopped - call ended")
                    break
                
                call_info = self.bridge.active_calls.get(call_id)
                if not call_info:
                    break
                
                # AUDIO RECOVERY: Check for stuck audio states
                if call_info.get('audio_sending_in_progress', False):
                    # Check if audio has been stuck for too long (30 seconds)
                    tts_task = call_info.get('tts_playback_task')
                    if tts_task and not tts_task.done():
                        # Check if task has been running for too long
                        task_start_time = getattr(tts_task, '_start_time', time.time())
                        if time.time() - task_start_time > 30:
                            logger.warning(f"[{call_id}] 🚨 AUDIO RECOVERY: TTS task stuck for >30s, forcing cancellation")
                            tts_task.cancel()
                            call_info['tts_playback_task'] = None
                            call_info['audio_sending_in_progress'] = False
                            call_info['audio_streaming_complete'] = True
                            call_info['audio_streaming_failed'] = True
                
                # CONTINUOUS TEST TONE INJECTION: Send test tones every 60 seconds (reduced frequency)
                call_duration = time.time() - call_info['start_time']
                time_since_last_tone = time.time() - call_info.get('last_test_tone_time', 0)
                
                if call_duration > 60 and time_since_last_tone > 60:  # Start after 60s, then every 60s
                    writer = call_info['vicial_writer']
                    await self._inject_mid_call_test_tone(call_id, writer, "beep")
                
                # Check for activity timeout (10 minutes of no audio)
                time_since_activity = time.time() - call_info['last_activity']
                if time_since_activity > 600:
                    logger.warning(f"[{call_id}] Call inactive for {time_since_activity:.1f}s - ending call")
                    break
                
                # Check for maximum call duration (20 minutes)
                if call_duration > 1200:
                    logger.info(f"[{call_id}] Call duration exceeded 20 minutes - ending call")
                    break
                    
        except Exception as e:
            logger.error(f"[{call_id}] Error in call monitoring: {e}")
    
    async def _cleanup_call(self, call_id: str, writer):
        """Clean up call resources and save recordings"""
        # IMMEDIATE EXIT: Check if call is already cleaned up
        if call_id not in self.bridge.active_calls:
            logger.debug(f"[{call_id}] Call already cleaned up, skipping")
            return
        
        call_info = self.bridge.active_calls[call_id]
        call_duration = time.time() - call_info['start_time']
        logger.info(f"[{call_id}] Cleaning up call (duration: {call_duration:.1f}s)")
        
        # CRITICAL FIX: Wait for audio streaming to complete before cleanup
        audio_streaming_timeout = 10.0  # 10 seconds timeout
        audio_streaming_start = time.time()
        
        while not call_info.get('audio_streaming_complete', True) and not call_info.get('audio_streaming_failed', False):
            if time.time() - audio_streaming_start > audio_streaming_timeout:
                logger.warning(f"[{call_id}] Audio streaming timeout after {audio_streaming_timeout}s - proceeding with cleanup")
                break
            
            logger.debug(f"[{call_id}] Waiting for audio streaming to complete...")
            await asyncio.sleep(0.1)  # Wait 100ms before checking again
        
        if call_info.get('audio_streaming_complete', False):
            logger.info(f"[{call_id}] ✅ Audio streaming completed successfully")
        elif call_info.get('audio_streaming_failed', False):
            logger.warning(f"[{call_id}] ⚠️ Audio streaming failed - proceeding with cleanup")
        
        # Check if call was already ended by immediate cleanup
        call_ended_by_immediate = call_info.get('call_ended', False)
        if call_ended_by_immediate:
            logger.info(f"[{call_id}] Call was ended by immediate cleanup - proceeding with recording save")
            
        call_duration = time.time() - call_info['start_time']
        
        logger.info(f"[{call_id}] Cleaning up call (duration: {call_duration:.1f}s)")
        
        # COMPLETE CALL RECORDING: Save the complete call recording
        call_recorder = call_info.get('call_recorder')
        if call_recorder:
            try:
                logger.info(f"[{call_id}] 🎙️ Starting call recording save process...")
                
                # Save complete call recording
                recording_file = await call_recorder.save_complete_call_recording()
                if recording_file:
                    logger.info(f"[{call_id}] ✅ Complete call recording saved: {recording_file}")
                else:
                    logger.warning(f"[{call_id}] ⚠️ No complete call recording was saved (no audio data)")
                
                # Save separate audio tracks for detailed analysis
                separate_tracks = await call_recorder.save_separate_audio_tracks()
                if separate_tracks:
                    logger.info(f"[{call_id}] ✅ Separate audio tracks saved: {list(separate_tracks.keys())}")
                else:
                    logger.info(f"[{call_id}] ℹ️ No separate audio tracks to save")
                
                # Log recording statistics
                recording_stats = call_recorder.get_stats()
                logger.info(f"[{call_id}] 📊 Recording stats: {recording_stats['duration_ms']:.1f}ms, "
                           f"{recording_stats['vicial_audio_chunks']} caller chunks, "
                           f"{recording_stats['ai_audio_chunks']} AI chunks")
                
                # CRITICAL: Log if no audio was recorded at all
                if recording_stats['vicial_audio_chunks'] == 0 and recording_stats['ai_audio_chunks'] == 0:
                    logger.warning(f"[{call_id}] ⚠️ WARNING: No audio was recorded during this call!")
                elif recording_stats['vicial_audio_chunks'] == 0:
                    logger.warning(f"[{call_id}] ⚠️ WARNING: No caller audio was recorded!")
                elif recording_stats['ai_audio_chunks'] == 0:
                    logger.warning(f"[{call_id}] ⚠️ WARNING: No AI audio was recorded!")
                
            except Exception as e:
                logger.error(f"[{call_id}] ❌ CRITICAL ERROR saving call recording: {e}")
                import traceback
                logger.error(f"[{call_id}] Traceback: {traceback.format_exc()}")
        else:
            logger.info(f"[{call_id}] ℹ️ Call recording disabled - no recording to save")
        
        # Close Local Model connection if not already closed by immediate cleanup
        if not call_ended_by_immediate and 'local_model_client' in call_info and call_info['local_model_client']:
            try:
                await call_info['local_model_client'].close_call()
            except Exception as e:
                logger.error(f"[{call_id}] Error closing Local Model connection: {e}")
        
        # Log call statistics
        frames_sent = call_info.get('audio_frames_sent', 0)
        frames_received = call_info.get('audio_frames_received', 0)
        ai_responses = call_info.get('ai_responses_received', 0)
        tts_requests = call_info.get('tts_requests_sent', 0)
        tts_responses = call_info.get('tts_responses_received', 0)
        buffer_size = len(call_info.get('audio_buffer', []))
        vad_speech_segments = call_info.get('vad_speech_segments', 0)
        vad_silence_discarded = call_info.get('vad_silence_discarded', 0)
        
        logger.info(f"[{call_id}] Call stats: {frames_sent} frames sent, {frames_received} frames received")
        logger.info(f"[{call_id}] AI stats: {ai_responses} responses, {tts_requests} TTS requests, {tts_responses} TTS responses")
        logger.info(f"[{call_id}] VAD stats: {vad_speech_segments} speech segments, {vad_silence_discarded} silence chunks discarded")
        logger.info(f"[{call_id}] Buffer: {buffer_size} bytes")
        
        # Update global VAD statistics
        self.bridge.stats['vad_speech_segments'] += vad_speech_segments
        self.bridge.stats['vad_silence_discarded'] += vad_silence_discarded
        
        # Remove from active calls (defensive check)
        if call_id in self.bridge.active_calls:
            del self.bridge.active_calls[call_id]
            self.bridge.stats['active_calls'] -= 1
        
        if frames_sent > 0 or frames_received > 0:
            self.bridge.stats['successful_calls'] += 1
        else:
            self.bridge.stats['failed_calls'] += 1
        
        # Close Vicial connection if not already closed by immediate cleanup
        if not call_ended_by_immediate:
            try:
                if writer and not writer.is_closing():
                    writer.close()
                    await writer.wait_closed()
            except Exception as e:
                logger.error(f"[{call_id}] Error closing Vicial connection: {e}")
        
        # FINAL SAFETY CHECK: Ensure call recording was saved
        if call_recorder:
            try:
                # Double-check if recording was actually saved
                recording_stats = call_recorder.get_stats()
                if recording_stats['vicial_audio_chunks'] > 0 or recording_stats['ai_audio_chunks'] > 0:
                    # Try to save again if it wasn't saved before
                    recording_file = await call_recorder.save_complete_call_recording()
                    if recording_file:
                        logger.info(f"[{call_id}] 🔄 Backup call recording saved: {recording_file}")
                    else:
                        logger.warning(f"[{call_id}] ⚠️ Backup call recording save failed")
            except Exception as e:
                logger.error(f"[{call_id}] Error in backup call recording save: {e}")
        
        logger.info(f"[{call_id}] ✅ Call cleanup completed successfully")

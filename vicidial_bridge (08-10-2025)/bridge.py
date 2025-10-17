"""
Main bridge module for Vicidial Bridge

Contains the VicialLocalModelBridge class that orchestrates the entire voice bot system.
"""

import asyncio
import logging
import os
import struct
import time
import numpy as np
from array import array
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

try:
    from .config import (
        VICIAL_AUDIOSOCKET_HOST, VICIAL_AUDIOSOCKET_PORT, LOCAL_MODEL_WS_URL,
        TARGET_EXTENSION, VICIAL_SAMPLE_RATE, LOCAL_MODEL_SAMPLE_RATE,
        AUDIO_PROCESS_INTERVAL, MIN_AUDIO_CHUNK_SIZE, MAX_AUDIO_CHUNK_SIZE,
        VAD_ENABLED, VAD_ENERGY_THRESHOLD, VAD_SILENCE_DURATION_MS, VAD_MIN_SPEECH_DURATION_MS,
        VAD_DEBUG_LOGGING, VAD_HIGH_PASS_CUTOFF, VAD_MIN_CONSECUTIVE_FRAMES, VAD_SPECTRAL_FLATNESS_THRESHOLD,
        VAD_BARGE_IN_CONSECUTIVE_FRAMES, VAD_BARGE_IN_ENABLED,
        SAVE_COMPLETE_CALLS, SAVE_AI_AUDIO, AI_AUDIO_DIR, validate_environment
    )
    from .audio_processing import AudioProcessor, AudioSocketProtocol
    from .voice_activity_detection import VoiceActivityDetector
    from .call_recording import CallRecorder
    from .websocket_client import LocalModelWebSocketClient
    from .bridge_methods import BridgeMethods
except ImportError:
    from config import (
        VICIAL_AUDIOSOCKET_HOST, VICIAL_AUDIOSOCKET_PORT, LOCAL_MODEL_WS_URL,
        TARGET_EXTENSION, VICIAL_SAMPLE_RATE, LOCAL_MODEL_SAMPLE_RATE,
        AUDIO_PROCESS_INTERVAL, MIN_AUDIO_CHUNK_SIZE, MAX_AUDIO_CHUNK_SIZE,
        VAD_ENABLED, VAD_ENERGY_THRESHOLD, VAD_SILENCE_DURATION_MS, VAD_MIN_SPEECH_DURATION_MS,
        VAD_DEBUG_LOGGING, VAD_HIGH_PASS_CUTOFF, VAD_MIN_CONSECUTIVE_FRAMES, VAD_SPECTRAL_FLATNESS_THRESHOLD,
        VAD_BARGE_IN_CONSECUTIVE_FRAMES, VAD_BARGE_IN_ENABLED,
        SAVE_COMPLETE_CALLS, SAVE_AI_AUDIO, AI_AUDIO_DIR, validate_environment
    )
    from audio_processing import AudioProcessor, AudioSocketProtocol
    from voice_activity_detection import VoiceActivityDetector
    from call_recording import CallRecorder
    from websocket_client import LocalModelWebSocketClient
    from bridge_methods import BridgeMethods

logger = logging.getLogger(__name__)

class VicialLocalModelBridge:
    """
    Main bridge class that handles connections from Vicial and routes them to local model server
    Manages audio processing, WebSocket communication, and call lifecycle with optimized latency
    """
    
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.active_calls: Dict[str, Dict[str, Any]] = {}
        self.server = None
        self.stats = {
            'total_calls': 0,
            'active_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'vad_speech_segments': 0,
            'vad_silence_discarded': 0,
            'start_time': time.time()
        }
        # Initialize helper methods
        self.methods = BridgeMethods(self)
        logger.info("VicialLocalModelBridge initialized")
    
    async def handle_vicial_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """
        Handle incoming connection from Vicial AudioSocket with proper protocol framing
        This is called when someone dials extension 8888
        """
        client_addr = writer.get_extra_info('peername')
        call_id = f"{client_addr[0]}:{client_addr[1]}:{int(time.time())}"
        call_uuid = None
        
        logger.info(f"[{call_id}] New call from Vicial: {client_addr}")
        
        # Update statistics
        self.stats['total_calls'] += 1
        self.stats['active_calls'] += 1
        
        # Create local model client
        local_model_client = LocalModelWebSocketClient()
        
        # Check local model server health before proceeding
        if not await local_model_client.health_check():
            logger.error(f"[{call_id}] Local model server is not accessible - rejecting call")
            await self.methods._send_error_tone(writer)
            await self.methods._cleanup_call(call_id, writer)
            return
        
        # Start local model call
        if not await local_model_client.start_call():
            logger.error(f"[{call_id}] Failed to start local model call - rejecting")
            await self.methods._send_error_tone(writer)
            await self.methods._cleanup_call(call_id, writer)
            return
        
        # Initialize VAD for this call
        vad = None
        if VAD_ENABLED:
            vad = VoiceActivityDetector(
                energy_threshold=VAD_ENERGY_THRESHOLD,
                silence_duration_ms=VAD_SILENCE_DURATION_MS,
                min_speech_duration_ms=VAD_MIN_SPEECH_DURATION_MS,
                sample_rate=VICIAL_SAMPLE_RATE,
                high_pass_cutoff=VAD_HIGH_PASS_CUTOFF,
                min_consecutive_frames=VAD_MIN_CONSECUTIVE_FRAMES,
                spectral_flatness_threshold=VAD_SPECTRAL_FLATNESS_THRESHOLD
            )
            # Configure barge-in detection threshold
            if VAD_BARGE_IN_ENABLED:
                vad.barge_in_consecutive_frames_threshold = VAD_BARGE_IN_CONSECUTIVE_FRAMES
                logger.info(f"[{call_id}] Enhanced VAD enabled: threshold={VAD_ENERGY_THRESHOLD}, "
                           f"silence={VAD_SILENCE_DURATION_MS}ms, min_speech={VAD_MIN_SPEECH_DURATION_MS}ms, "
                           f"high_pass={VAD_HIGH_PASS_CUTOFF}Hz, consecutive_frames={VAD_MIN_CONSECUTIVE_FRAMES}, "
                           f"barge_in_frames={VAD_BARGE_IN_CONSECUTIVE_FRAMES}")
            else:
                logger.info(f"[{call_id}] Enhanced VAD enabled: threshold={VAD_ENERGY_THRESHOLD}, "
                           f"silence={VAD_SILENCE_DURATION_MS}ms, min_speech={VAD_MIN_SPEECH_DURATION_MS}ms, "
                           f"high_pass={VAD_HIGH_PASS_CUTOFF}Hz, consecutive_frames={VAD_MIN_CONSECUTIVE_FRAMES}")
        else:
            logger.info(f"[{call_id}] VAD disabled - processing all audio chunks")

        # Initialize call recorder for complete call recording
        call_recorder = CallRecorder(call_id) if SAVE_COMPLETE_CALLS else None
        
        # Store call information with optimized audio buffering and VAD
        self.active_calls[call_id] = {
            'vicial_reader': reader,
            'vicial_writer': writer,
            'local_model_client': local_model_client,
            'vad': vad,                            # Voice Activity Detector
            'call_recorder': call_recorder,        # Complete call recording
            'start_time': time.time(),
            'audio_frames_sent': 0,
            'audio_frames_received': 0,
            'ai_responses_received': 0,            # Track AI responses
            'tts_requests_sent': 0,                # Track TTS requests
            'tts_responses_received': 0,           # Track TTS responses
            'last_activity': time.time(),
            'audio_buffer': bytearray(),           # Audio buffer for streaming
            'processing_audio': False,             # Audio processing flag
            'call_uuid': None,                     # Store AudioSocket UUID
            'test_tones_sent': 0,                  # Track test tones sent
            'last_test_tone_time': 0,              # Track last test tone time
            'vad_speech_segments': 0,              # Track VAD speech segments
            'vad_silence_discarded': 0,            # Track VAD silence discarded
            'processing_tasks': [],                # Track processing tasks for cancellation
            'call_ended': False,                   # Flag to indicate call has ended
            # BARGE-IN SUPPORT: Turn management and TTS control
            'current_turn_id': 0,                  # Current conversation turn
            'tts_playback_task': None,             # Currently running TTS playback task
            'audio_sending_in_progress': False,    # Flag to prevent race conditions
            'audio_streaming_complete': True,      # Flag to track audio completion
            'audio_streaming_failed': False,       # Flag to track audio failures
            'barge_in_count': 0,                   # Count of barge-in events
            'last_barge_in_time': 0                # Timestamp of last barge-in
        }
        
        logger.info(f"[{call_id}] Call setup complete - starting audio processing")
        
        # DEBUG: Inject test audio to verify AudioSocket is working
        await self.methods._inject_test_audio(call_id, writer)
        
        try:
            # Start bidirectional audio processing with task tracking
            processing_tasks = [
                asyncio.create_task(self._process_vicial_audio(call_id)),
                asyncio.create_task(self.methods._process_local_model_responses(call_id)),
                asyncio.create_task(self.methods._monitor_call_health(call_id))
            ]
            
            # Store tasks for potential cancellation
            self.active_calls[call_id]['processing_tasks'] = processing_tasks
            
            # Wait for all tasks to complete
            await asyncio.gather(*processing_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"[{call_id}] Error during call processing: {e}")
        finally:
            # CRITICAL: Always cleanup call and save recordings, regardless of how call ended
            # This will run even after immediate cleanup
            if call_id in self.active_calls:
                logger.info(f"[{call_id}] Call processing ended - starting final cleanup and recording save")
                await self.methods._cleanup_call(call_id, writer)
            else:
                logger.warning(f"[{call_id}] Call processing ended - call was already removed from active_calls!")
    
    async def _process_vicial_audio(self, call_id: str):
        """Process audio from Vicial by buffering and sending to Local Model"""
        call_info = self.active_calls.get(call_id)
        if not call_info:
            return
        
        reader = call_info['vicial_reader']
        local_model_client = call_info['local_model_client']
        
        logger.info(f"[{call_id}] Starting Vicial audio processing")
        
        # LATENCY OPTIMIZATION: Use optimized audio processing parameters
        # Process audio every 50ms for near real-time streaming instead of 200ms
        PROCESS_INTERVAL = AUDIO_PROCESS_INTERVAL
        MIN_BUFFER_SIZE = MIN_AUDIO_CHUNK_SIZE
        MAX_BUFFER_SIZE = MAX_AUDIO_CHUNK_SIZE
        
        try:
            frame_count = 0
            last_process_time = time.time()
            
            # First, read UUID message from AudioSocket
            logger.info(f"[{call_id}] Waiting for AudioSocket UUID...")
            msg_type, payload = await AudioSocketProtocol.read_frame(reader)
            if msg_type == AudioSocketProtocol.TYPE_UUID and len(payload) == 16:
                call_uuid = payload.hex()
                call_info['call_uuid'] = call_uuid  # Store UUID in call info
                logger.info(f"[{call_id}] AudioSocket UUID received: {call_uuid[:8]}...")
            else:
                logger.warning(f"[{call_id}] Unexpected first message: type={msg_type}, length={len(payload) if payload else 0}")
            
            while call_id in self.active_calls:
                frame_count += 1
                
                # Read framed audio data from AudioSocket
                msg_type, payload = await AudioSocketProtocol.read_frame(reader)
                
                if msg_type is None:  # Connection closed
                    logger.info(f"[{call_id}] AudioSocket connection closed - call ended naturally")
                    await self._immediate_call_cleanup(call_id)
                    break
                elif msg_type == AudioSocketProtocol.TYPE_HANGUP:
                    logger.info(f"[{call_id}] AudioSocket hangup received - call ended naturally")
                    await self._immediate_call_cleanup(call_id)
                    break
                elif msg_type == AudioSocketProtocol.TYPE_AUDIO:
                    audio_data = payload
                elif msg_type == AudioSocketProtocol.TYPE_DTMF:
                    logger.debug(f"[{call_id}] DTMF received: {payload}")
                    continue  # Skip DTMF for now
                else:
                    logger.debug(f"[{call_id}] Unknown AudioSocket message type: {msg_type}")
                    continue
                
                if not audio_data:
                    continue
                
                # Update activity timestamp
                call_info['last_activity'] = time.time()
                call_info['audio_frames_sent'] += 1
                
                # Process audio from Vicial
                logger.info(f"[{call_id}] 📥 Received {len(audio_data)} bytes of audio from Vicial")
                
                # COMPLETE CALL RECORDING: Add Vicial audio to call recorder
                call_recorder = call_info.get('call_recorder')
                if call_recorder:
                    call_recorder.add_vicial_audio(audio_data)
                
                # CRITICAL DEBUG: Save raw incoming audio for analysis
                await self.audio_processor.save_debug_audio(call_id, audio_data, "vicial_raw_8k", VICIAL_SAMPLE_RATE)
                
                # Log audio characteristics for debugging
                samples = len(audio_data) // 2
                duration_ms = (samples / VICIAL_SAMPLE_RATE) * 1000
                max_amplitude = max(abs(s) for s in struct.unpack(f'<{samples}h', audio_data)) if samples > 0 else 0
                logger.info(f"[{call_id}] 📊 Audio stats: {samples} samples, {duration_ms:.1f}ms, max_amp={max_amplitude}")
                
                # CRITICAL DEBUG: Analyze first 20 samples of raw audio
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                first_20_samples = audio_array[:20].tolist()
                logger.info(f"[{call_id}] 🔍 Raw Vicial audio first 20 samples: {first_20_samples}")
                
                # VOICE ACTIVITY DETECTION: Process audio through VAD
                vad = call_info.get('vad')
                if vad:
                    # REAL-TIME BARGE-IN: Check for speech onset immediately on each chunk
                    if VAD_BARGE_IN_ENABLED:
                        is_speech_now = vad.is_speech_for_barge_in(audio_data)
                    else:
                        is_speech_now = vad.is_speech(audio_data)
                    
                    # If speech just started and TTS is playing, cancel TTS immediately (real-time barge-in)
                    # Add debounce to prevent false positives from brief noise
                    current_time = time.time()
                    time_since_last_barge_in = current_time - call_info.get('last_barge_in_time', 0)
                    
                    if (is_speech_now and 
                        call_info.get('tts_playback_task') and 
                        not call_info['tts_playback_task'].done() and
                        time_since_last_barge_in > 1.0):  # 1 second debounce
                        
                        call_info['barge_in_count'] += 1
                        call_info['last_barge_in_time'] = current_time
                        logger.info(f"[{call_id}] 🚨 REAL-TIME BARGE-IN: User started speaking, cancelling AI playback immediately")
                        
                        # Cancel current TTS playback immediately
                        call_info['tts_playback_task'].cancel()
                        call_info['tts_playback_task'] = None
                        
                        # Increment turn ID to invalidate any pending responses
                        call_info['current_turn_id'] += 1
                        new_turn_id = call_info['current_turn_id']
                        
                        logger.info(f"[{call_id}] 🔄 Turn ID incremented to {new_turn_id} - old responses will be ignored")
                        
                        # Clear any pending audio buffers
                        call_info['audio_buffer'] = bytearray()
                        call_info['audio_sending_in_progress'] = False
                        call_info['audio_streaming_complete'] = True
                        
                        # Send a brief silence to flush the audio buffer
                        await self._send_silence_flush(call_id, call_info['vicial_writer'])
                    
                    # Process audio chunk through VAD for complete speech segments
                    speech_segment = vad.process_audio_chunk(audio_data)
                    
                    if speech_segment:
                        # VAD detected complete speech segment
                        call_info['vad_speech_segments'] += 1
                        logger.info(f"[{call_id}] 🎤 VAD detected speech segment #{call_info['vad_speech_segments']}: "
                                   f"{len(speech_segment)} bytes")
                        
                        # Upsample speech segment to 16kHz
                        upsampled_audio = self.audio_processor.upsample_8k_to_16k(speech_segment)
                        if upsampled_audio:
                            # CRITICAL DEBUG: Save upsampled speech segment for analysis
                            await self.audio_processor.save_debug_audio(call_id, upsampled_audio, "vad_speech_16k", LOCAL_MODEL_SAMPLE_RATE)
                            
                            # Send complete speech segment to Local Model
                            asyncio.create_task(self.methods._stream_audio_chunk(call_id, upsampled_audio))
                        else:
                            logger.warning(f"[{call_id}] ⚠️ Upsampling VAD speech segment returned empty audio!")
                    else:
                        # VAD filtered out silence/noise
                        call_info['vad_silence_discarded'] += 1
                        if VAD_DEBUG_LOGGING:
                            logger.debug(f"[{call_id}] 🔇 VAD filtered silence #{call_info['vad_silence_discarded']} "
                                       f"(total discarded: {call_info['vad_silence_discarded']})")
                else:
                    # VAD disabled - process all audio chunks (original behavior)
                    logger.debug(f"[{call_id}] VAD disabled - processing all audio chunks")
                    
                    # Upsample to 16kHz and add to audio buffer
                    upsampled_audio = self.audio_processor.upsample_8k_to_16k(audio_data)
                    if upsampled_audio:
                        call_info['audio_buffer'].extend(upsampled_audio)
                        logger.info(f"[{call_id}] 📥 Added {len(upsampled_audio)} bytes to buffer (total: {len(call_info['audio_buffer'])} bytes)")
                        
                        # CRITICAL DEBUG: Save upsampled audio for analysis
                        await self.audio_processor.save_debug_audio(call_id, upsampled_audio, "upsampled_16k", LOCAL_MODEL_SAMPLE_RATE)
                    else:
                        logger.warning(f"[{call_id}] ⚠️ Upsampling returned empty audio!")
                    
                    # LATENCY OPTIMIZATION: Stream audio as soon as minimum buffer is reached
                    # Instead of waiting for fixed intervals, send audio immediately when ready
                    current_time = time.time()
                    buffer_size = len(call_info['audio_buffer'])
                    
                    # Send audio immediately if we have enough data, or wait for process interval
                    should_process = (
                        buffer_size >= MIN_BUFFER_SIZE or  # Send immediately when minimum reached
                        (current_time - last_process_time >= PROCESS_INTERVAL and buffer_size > 0) or  # Or at interval
                        buffer_size >= MAX_BUFFER_SIZE  # Or when buffer is getting too large
                    )
                    
                    if should_process and not call_info['processing_audio']:
                        # Mark as processing to prevent concurrent processing
                        call_info['processing_audio'] = True
                        last_process_time = current_time
                        
                        # Extract audio chunk to process (don't exceed max buffer size)
                        chunk_size = min(buffer_size, MAX_BUFFER_SIZE)
                        audio_chunk = bytes(call_info['audio_buffer'][:chunk_size])
                        call_info['audio_buffer'] = call_info['audio_buffer'][chunk_size:]
                        
                        # Calculate actual audio duration for logging
                        audio_duration_ms = (len(audio_chunk) / (LOCAL_MODEL_SAMPLE_RATE * 2)) * 1000
                        
                        logger.debug(f"[{call_id}] 🔄 Streaming {len(audio_chunk)} bytes ({audio_duration_ms:.1f}ms) to Local Model")
                        
                        # CRITICAL DEBUG: Save audio chunk being sent to Ultravox
                        await self.audio_processor.save_debug_audio(call_id, audio_chunk, "sent_to_ultravox_16k", LOCAL_MODEL_SAMPLE_RATE)
                        
                        # LATENCY OPTIMIZATION: Send audio to Local Model immediately (non-blocking)
                        # Use create_task for concurrent processing without blocking the main loop
                        asyncio.create_task(self.methods._stream_audio_chunk(call_id, audio_chunk))
                    
        except asyncio.TimeoutError:
            logger.debug(f"[{call_id}] Audio read timeout (normal)")
        except Exception as e:
            logger.error(f"[{call_id}] Error in audio processing: {e}")
        finally:
            # VAD CLEANUP: Flush any remaining speech from VAD buffer
            if call_id in self.active_calls:
                call_info = self.active_calls[call_id]
                vad = call_info.get('vad')
                if vad:
                    remaining_speech = vad.flush_remaining_speech()
                    if remaining_speech:
                        logger.info(f"[{call_id}] 🎤 VAD flushed remaining speech: {len(remaining_speech)} bytes")
                        
                        # Upsample and send remaining speech
                        upsampled_audio = self.audio_processor.upsample_8k_to_16k(remaining_speech)
                        if upsampled_audio:
                            asyncio.create_task(self.methods._stream_audio_chunk(call_id, upsampled_audio))
            
            # Check if call was already cleaned up by immediate cleanup
            if call_id in self.active_calls and not self.active_calls[call_id].get('call_ended', False):
                logger.info(f"[{call_id}] Audio processing ended - cleanup will be handled by main handler")
            else:
                logger.info(f"[{call_id}] Audio processing ended - call already cleaned up")
    
    async def start_server(self):
        """Start the AudioSocket server for Vicial connections"""
        logger.info("Starting Vicial-Local Model bridge server...")
        
        # Validate environment configuration
        try:
            validate_environment()
        except ValueError as e:
            logger.error(str(e))
            return False
        
        # Test local model server connectivity
        test_client = LocalModelWebSocketClient()
        if not await test_client.health_check():
            logger.error("Cannot start bridge - Local model server is not accessible")
            logger.error(f"Please check your local model server at {LOCAL_MODEL_HOST}:{LOCAL_MODEL_PORT}")
            return False
        
        try:
            self.server = await asyncio.start_server(
                self.handle_vicial_connection,
                VICIAL_AUDIOSOCKET_HOST,
                VICIAL_AUDIOSOCKET_PORT
            )
            
            addr = self.server.sockets[0].getsockname()
            logger.info("=" * 80)
            logger.info("VICIAL-LOCAL MODEL BRIDGE SERVER STARTED")
            logger.info("=" * 80)
            logger.info(f"AudioSocket listening on: {addr[0]}:{addr[1]}")
            logger.info(f"Target extension: {TARGET_EXTENSION}")
            logger.info(f"Local Model Server: {LOCAL_MODEL_WS_URL}")
            logger.info("LATENCY OPTIMIZATION: 50ms audio processing for near real-time streaming")
            logger.info("TTS ENGINE: Piper neural voices for high-quality speech synthesis")
            logger.info("Environment: Configuration loaded from .env file")
            logger.info("Ready to handle voice calls!")
            logger.info("=" * 80)
            
            # Start statistics logging
            asyncio.create_task(self._log_statistics())
            
            # Wait for server to be closed
            async with self.server:
                await self.server.serve_forever()
                
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Server error: {e}")
            return False
        finally:
            await self._shutdown()
    
    async def _log_statistics(self):
        """Periodically log server statistics"""
        while True:
            try:
                await asyncio.sleep(60)  # Log every minute
                
                uptime = time.time() - self.stats['start_time']
                audio_stats = self.audio_processor.get_stats()
                
                logger.info("=" * 50)
                logger.info("BRIDGE STATISTICS")
                logger.info("=" * 50)
                logger.info(f"Uptime: {uptime/3600:.1f} hours")
                logger.info(f"Total calls: {self.stats['total_calls']}")
                logger.info(f"Active calls: {self.stats['active_calls']}")
                logger.info(f"Successful calls: {self.stats['successful_calls']}")
                logger.info(f"Failed calls: {self.stats['failed_calls']}")
                logger.info(f"Audio frames processed: {audio_stats['frames_upsampled']} up, {audio_stats['frames_downsampled']} down")
                logger.info(f"Audio data processed: {audio_stats['total_input_bytes']/1024/1024:.1f}MB in, {audio_stats['total_output_bytes']/1024/1024:.1f}MB out")
                logger.info(f"VAD processing: {self.stats['vad_speech_segments']} speech segments, {self.stats['vad_silence_discarded']} silence chunks filtered")
                if audio_stats['errors'] > 0:
                    logger.warning(f"Audio processing errors: {audio_stats['errors']}")
                logger.info("=" * 50)
                
            except Exception as e:
                logger.error(f"Error logging statistics: {e}")
    
    async def _shutdown(self):
        """Clean shutdown of the bridge"""
        logger.info("Shutting down Vicial-Local Model bridge...")
        
        # Close all active calls
        for call_id in list(self.active_calls.keys()):
            logger.info(f"Closing active call: {call_id}")
            await self.methods._cleanup_call(call_id, self.active_calls[call_id]['vicial_writer'])
        
        # Close server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        logger.info("Bridge shutdown complete")
    
    async def _immediate_call_cleanup(self, call_id: str):
        """Perform immediate cleanup when AudioSocket disconnects - but preserve call for recording save"""
        call_info = self.active_calls.get(call_id)
        if not call_info or call_info.get('call_ended', False):
            logger.debug(f"[{call_id}] Call already cleaned up or ended, skipping immediate cleanup")
            return
        
        logger.info(f"[{call_id}] 🚨 IMMEDIATE CLEANUP: AudioSocket disconnected, marking call as ended")
        
        # Mark call as ended to prevent further processing
        call_info['call_ended'] = True
        
        # Cancel all processing tasks for this call
        processing_tasks = call_info.get('processing_tasks', [])
        for task in processing_tasks:
            if not task.done() and task != asyncio.current_task():
                logger.debug(f"[{call_id}] Cancelling task: {task.get_name()}")
                task.cancel()
        
        # Close WebSocket connection immediately to stop further processing
        local_model_client = call_info.get('local_model_client')
        if local_model_client:
            try:
                await local_model_client.close_call()
                logger.debug(f"[{call_id}] Closed WebSocket connection")
            except Exception as e:
                logger.error(f"[{call_id}] Error closing WebSocket: {e}")
        
        # Close the AudioSocket writer to prevent broken pipe errors
        writer = call_info.get('vicial_writer')
        if writer and not writer.is_closing():
            try:
                writer.close()
                await writer.wait_closed()
                logger.debug(f"[{call_id}] Closed AudioSocket writer")
            except Exception as e:
                logger.debug(f"[{call_id}] Error closing writer (expected): {e}")
        
        # DO NOT call _cleanup_call() here - let the main handler do it
        # DO NOT remove from active_calls - let the main handler do it
        logger.info(f"[{call_id}] ✅ Immediate cleanup completed - call marked as ended, main cleanup will save recording")
    
    async def _send_silence_flush(self, call_id: str, writer):
        """Send a brief silence to flush the audio buffer after barge-in"""
        try:
            # Send 100ms of silence to flush any remaining audio
            silence_duration_ms = 100
            silence_samples = int(VICIAL_SAMPLE_RATE * silence_duration_ms / 1000)
            silence_bytes = b'\x00' * (silence_samples * 2)  # 16-bit samples
            
            logger.debug(f"[{call_id}] 🔇 Sending {silence_duration_ms}ms silence flush...")
            success = await AudioSocketProtocol.write_audio_frame(writer, silence_bytes)
            
            if success:
                logger.debug(f"[{call_id}] ✅ Silence flush sent successfully")
            else:
                logger.warning(f"[{call_id}] ⚠️ Failed to send silence flush")
                
        except Exception as e:
            logger.error(f"[{call_id}] Error sending silence flush: {e}")

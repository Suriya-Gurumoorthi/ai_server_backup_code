"""
Audio processing module for Vicidial Bridge

Handles audio format conversion, AudioSocket protocol, and audio processing utilities.
"""

import asyncio
import logging
import struct
import time
import numpy as np
import wave
import io
from datetime import datetime
from typing import Dict, Any, Optional

try:
    from .config import (
        VICIAL_SAMPLE_RATE, LOCAL_MODEL_SAMPLE_RATE, UPSAMPLE_FACTOR, DOWNSAMPLE_FACTOR,
        SAVE_DEBUG_AUDIO, DEBUG_AUDIO_DIR, AUDIO_CHUNK_SIZE, AUDIO_SILENCE_PADDING_MS, AUDIO_TIMING_TOLERANCE_MS,
        AUDIO_SPEED_MULTIPLIER
    )
except ImportError:
    from config import (
        VICIAL_SAMPLE_RATE, LOCAL_MODEL_SAMPLE_RATE, UPSAMPLE_FACTOR, DOWNSAMPLE_FACTOR,
        SAVE_DEBUG_AUDIO, DEBUG_AUDIO_DIR, AUDIO_CHUNK_SIZE, AUDIO_SILENCE_PADDING_MS, AUDIO_TIMING_TOLERANCE_MS,
        AUDIO_SPEED_MULTIPLIER
    )

logger = logging.getLogger(__name__)

# ==================== AUDIOSOCKET PROTOCOL ====================
class AudioSocketProtocol:
    """Handle Asterisk AudioSocket protocol framing"""
    
    # AudioSocket message types
    TYPE_HANGUP = 0x00
    TYPE_UUID = 0x01  
    TYPE_DTMF = 0x03
    TYPE_AUDIO = 0x10
    
    @staticmethod
    async def read_frame(reader: asyncio.StreamReader):
        """Read a framed message from AudioSocket"""
        try:
            # Read 3-byte header: type + length (big-endian)
            header = await reader.readexactly(3)
            if len(header) != 3:
                return None, None
                
            msg_type = header[0]
            payload_length = int.from_bytes(header[1:3], byteorder='big')
            
            # Read payload
            if payload_length > 0:
                payload = await reader.readexactly(payload_length)
                if len(payload) != payload_length:
                    return None, None
                return msg_type, payload
            else:
                return msg_type, b''
                
        except asyncio.IncompleteReadError:
            return None, None
        except Exception as e:
            logger.error(f"Error reading AudioSocket frame: {e}")
            return None, None
    
    @staticmethod
    def write_frame(writer: asyncio.StreamWriter, msg_type: int, payload: bytes):
        """Write a framed message to AudioSocket"""
        try:
            # Create 3-byte header: type + length (big-endian)
            header = bytes([msg_type]) + len(payload).to_bytes(2, byteorder='big')
            
            # Write header + payload
            writer.write(header + payload)
            return True
        except Exception as e:
            logger.error(f"Error writing AudioSocket frame: {e}")
            return False
    
    @staticmethod
    async def write_audio_frame(writer: asyncio.StreamWriter, audio_data: bytes):
        """Send audio data with proper AudioSocket framing (non-blocking)"""
        try:
            # Check if writer is still open
            if writer.is_closing():
                logger.debug("AudioSocket writer is closing, skipping audio frame")
                return False
            
            # Write the frame data
            AudioSocketProtocol.write_frame(writer, AudioSocketProtocol.TYPE_AUDIO, audio_data)
            
            # Use non-blocking drain with timeout to prevent blocking
            try:
                await asyncio.wait_for(writer.drain(), timeout=0.1)  # 100ms timeout
            except asyncio.TimeoutError:
                logger.debug("AudioSocket drain timeout - continuing without blocking")
                # Don't fail on timeout, just continue
            
            return True
        except (BrokenPipeError, ConnectionResetError) as e:
            logger.debug(f"AudioSocket connection broken: {e}")
            return False
        except Exception as e:
            logger.error(f"Error sending AudioSocket audio frame: {e}")
            return False
    
    @staticmethod
    async def write_audio_chunked(writer: asyncio.StreamWriter, audio_data: bytes, chunk_size: int = 320, sample_rate: int = 8000):
        """Send audio data in small chunks with real-time timing to prevent blocking and ensure complete delivery"""
        try:
            # Check if writer is still open
            if writer.is_closing():
                logger.debug("AudioSocket writer is closing, skipping chunked audio")
                return False
            
            # CRITICAL FIX: Validate PCM alignment to prevent corruption
            if len(audio_data) % 2 != 0:
                logger.warning("Audio data length is odd - padding with zero byte to maintain 16-bit alignment")
                audio_data = audio_data + b'\x00'
            
            # Ensure chunk size is aligned to 16-bit samples
            if chunk_size % 2 != 0:
                chunk_size = chunk_size - 1  # Make even
                logger.debug(f"Adjusted chunk size to {chunk_size} for 16-bit alignment")
            
            total_sent = 0
            chunk_count = 0
            total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
            
            # CRITICAL FIX: Calculate real-time audio duration per chunk with speed multiplier
            samples_per_chunk = chunk_size // 2  # 16-bit samples
            audio_ms_per_chunk = (samples_per_chunk / sample_rate) * 1000 / AUDIO_SPEED_MULTIPLIER  # milliseconds of audio per chunk (adjusted for speed)
            
            logger.info(f"🎵 Starting real-time audio delivery: {len(audio_data)} bytes in {total_chunks} chunks")
            logger.info(f"⏱️  Audio timing: {audio_ms_per_chunk:.1f}ms per chunk at {sample_rate}Hz (speed: {AUDIO_SPEED_MULTIPLIER}x)")
            
            # CRITICAL FIX: Add initial silence padding to help ViciDial lock to jitter buffer
            silence_duration_ms = AUDIO_SILENCE_PADDING_MS
            silence_samples = int(sample_rate * silence_duration_ms / 1000)
            silence_bytes = b'\x00' * (silence_samples * 2)  # 16-bit samples
            
            if silence_bytes:
                logger.debug(f"🔇 Sending {silence_duration_ms}ms silence padding...")
                success = await AudioSocketProtocol.write_audio_frame(writer, silence_bytes)
                if not success:
                    logger.error("Failed to send silence padding")
                    return False
                await asyncio.sleep(silence_duration_ms / 1000.0)  # Wait for silence to play
            
            # Track timing for real-time delivery
            start_time = time.monotonic()
            chunk_start_time = start_time
            
            # Send audio in chunks with proper real-time timing
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                
                # Send chunk with non-blocking framing
                success = await AudioSocketProtocol.write_audio_frame(writer, chunk)
                if not success:
                    logger.error(f"Failed to send audio chunk {chunk_count + 1}/{total_chunks} - connection may be closed")
                    return False
                
                total_sent += len(chunk)
                chunk_count += 1
                
                # CRITICAL FIX: Real-time timing - wait for actual audio duration
                now = time.monotonic()
                next_chunk_time = chunk_start_time + (audio_ms_per_chunk / 1000.0)
                sleep_time = next_chunk_time - now
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                elif sleep_time < -(AUDIO_TIMING_TOLERANCE_MS / 1000.0):  # If we're behind tolerance, log warning
                    logger.warning(f"⚠️ Audio timing lag: {abs(sleep_time)*1000:.1f}ms behind schedule (tolerance: {AUDIO_TIMING_TOLERANCE_MS}ms)")
                
                chunk_start_time += audio_ms_per_chunk / 1000.0
                
                # Log progress for long audio streams
                if chunk_count % 10 == 0 or chunk_count == total_chunks:
                    elapsed_time = time.monotonic() - start_time
                    expected_time = (chunk_count * audio_ms_per_chunk) / 1000.0
                    timing_ratio = elapsed_time / expected_time if expected_time > 0 else 1.0
                    logger.debug(f"📊 Audio progress: {chunk_count}/{total_chunks} chunks ({total_sent}/{len(audio_data)} bytes)")
                    logger.debug(f"⏱️  Timing: {elapsed_time:.2f}s elapsed, {expected_time:.2f}s expected (ratio: {timing_ratio:.3f})")
            
            # CRITICAL FIX: Ensure all data was sent
            if total_sent != len(audio_data):
                logger.error(f"Audio delivery incomplete: sent {total_sent}/{len(audio_data)} bytes")
                return False
            
            # Log final timing analysis
            total_elapsed = time.monotonic() - start_time
            total_audio_duration = (len(audio_data) // 2) / sample_rate
            timing_efficiency = total_audio_duration / total_elapsed if total_elapsed > 0 else 0
            
            logger.info(f"✅ Audio delivery complete: {chunk_count} chunks, {total_sent} bytes")
            logger.info(f"⏱️  Timing analysis: {total_elapsed:.2f}s elapsed for {total_audio_duration:.2f}s audio (efficiency: {timing_efficiency:.3f})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending chunked audio: {e}")
            return False
    
    @staticmethod
    async def write_audio_chunked_with_turn_control(writer: asyncio.StreamWriter, audio_data: bytes, 
                                                   chunk_size: int = 320, sample_rate: int = 8000, 
                                                   turn_validator=None):
        """Send audio data with turn-based validation to support barge-in interruption"""
        try:
            # Check if writer is still open
            if writer.is_closing():
                logger.debug("AudioSocket writer is closing, skipping turn-controlled audio")
                return False
            
            # CRITICAL FIX: Validate PCM alignment to prevent corruption
            if len(audio_data) % 2 != 0:
                logger.warning("Audio data length is odd - padding with zero byte to maintain 16-bit alignment")
                audio_data = audio_data + b'\x00'
            
            # Ensure chunk size is aligned to 16-bit samples
            if chunk_size % 2 != 0:
                chunk_size = chunk_size - 1  # Make even
                logger.debug(f"Adjusted chunk size to {chunk_size} for 16-bit alignment")
            
            total_sent = 0
            chunk_count = 0
            total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
            
            # CRITICAL FIX: Calculate real-time audio duration per chunk with speed multiplier
            samples_per_chunk = chunk_size // 2  # 16-bit samples
            audio_ms_per_chunk = (samples_per_chunk / sample_rate) * 1000 / AUDIO_SPEED_MULTIPLIER  # milliseconds of audio per chunk (adjusted for speed)
            
            logger.info(f"🎵 Starting turn-controlled audio delivery: {len(audio_data)} bytes in {total_chunks} chunks")
            logger.info(f"⏱️  Audio timing: {audio_ms_per_chunk:.1f}ms per chunk at {sample_rate}Hz (speed: {AUDIO_SPEED_MULTIPLIER}x)")
            
            # CRITICAL FIX: Add initial silence padding to help ViciDial lock to jitter buffer
            silence_duration_ms = AUDIO_SILENCE_PADDING_MS
            silence_samples = int(sample_rate * silence_duration_ms / 1000)
            silence_bytes = b'\x00' * (silence_samples * 2)  # 16-bit samples
            
            if silence_bytes:
                # Check turn validity before sending silence
                if turn_validator and not turn_validator():
                    logger.info("🚫 Turn validation failed - skipping silence padding")
                    return False
                
                logger.debug(f"🔇 Sending {silence_duration_ms}ms silence padding...")
                success = await AudioSocketProtocol.write_audio_frame(writer, silence_bytes)
                if not success:
                    logger.error("Failed to send silence padding")
                    return False
                # Remove the sleep to prevent blocking - let the audio timing handle it
                # await asyncio.sleep(silence_duration_ms / 1000.0)  # Wait for silence to play
            
            # Track timing for real-time delivery
            start_time = time.monotonic()
            chunk_start_time = start_time
            
            # Send audio in chunks with proper real-time timing and turn validation
            for i in range(0, len(audio_data), chunk_size):
                # BARGE-IN SUPPORT: Check turn validity before each chunk
                if turn_validator and not turn_validator():
                    logger.info(f"🚫 Turn validation failed at chunk {chunk_count + 1}/{total_chunks} - stopping audio delivery")
                    return False
                
                chunk = audio_data[i:i + chunk_size]
                
                # Send chunk with non-blocking framing
                success = await AudioSocketProtocol.write_audio_frame(writer, chunk)
                if not success:
                    logger.error(f"Failed to send audio chunk {chunk_count + 1}/{total_chunks} - connection may be closed")
                    return False
                
                total_sent += len(chunk)
                chunk_count += 1
                
                # CRITICAL FIX: Real-time timing - wait for actual audio duration
                now = time.monotonic()
                next_chunk_time = chunk_start_time + (audio_ms_per_chunk / 1000.0)
                sleep_time = next_chunk_time - now
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                elif sleep_time < -(AUDIO_TIMING_TOLERANCE_MS / 1000.0):  # If we're behind tolerance, log warning
                    logger.warning(f"⚠️ Audio timing lag: {abs(sleep_time)*1000:.1f}ms behind schedule (tolerance: {AUDIO_TIMING_TOLERANCE_MS}ms)")
                
                chunk_start_time += audio_ms_per_chunk / 1000.0
                
                # Log progress for long audio streams
                if chunk_count % 10 == 0 or chunk_count == total_chunks:
                    elapsed_time = time.monotonic() - start_time
                    expected_time = (chunk_count * audio_ms_per_chunk) / 1000.0
                    timing_ratio = elapsed_time / expected_time if expected_time > 0 else 1.0
                    logger.debug(f"📊 Audio progress: {chunk_count}/{total_chunks} chunks ({total_sent}/{len(audio_data)} bytes)")
                    logger.debug(f"⏱️  Timing: {elapsed_time:.2f}s elapsed, {expected_time:.2f}s expected (ratio: {timing_ratio:.3f})")
            
            # CRITICAL FIX: Ensure all data was sent
            if total_sent != len(audio_data):
                logger.error(f"Audio delivery incomplete: sent {total_sent}/{len(audio_data)} bytes")
                return False
            
            # Log final timing analysis
            total_elapsed = time.monotonic() - start_time
            total_audio_duration = (len(audio_data) // 2) / sample_rate
            timing_efficiency = total_audio_duration / total_elapsed if total_elapsed > 0 else 0
            
            logger.info(f"✅ Turn-controlled audio delivery complete: {chunk_count} chunks, {total_sent} bytes")
            logger.info(f"⏱️  Timing analysis: {total_elapsed:.2f}s elapsed for {total_audio_duration:.2f}s audio (efficiency: {timing_efficiency:.3f})")
            
            return True
            
        except asyncio.CancelledError:
            logger.info("Turn-controlled audio delivery was cancelled")
            raise
        except Exception as e:
            logger.error(f"Error sending turn-controlled audio: {e}")
            return False

# ==================== AUDIO PROCESSOR ====================
class AudioProcessor:
    """
    Handles high-quality audio processing for converting between Vicial (8kHz) and Local Model (16kHz)
    Uses linear interpolation for upsampling and averaging for downsampling
    """
    
    def __init__(self):
        logger.info("Initializing AudioProcessor with optimized NumPy operations for low latency")
        self.stats = {
            'frames_upsampled': 0,
            'frames_downsampled': 0,
            'total_input_bytes': 0,
            'total_output_bytes': 0,
            'errors': 0
        }
        
        # LATENCY OPTIMIZATION: Pre-allocate NumPy arrays for better performance
        self._temp_upsample_buffer = None
        self._temp_downsample_buffer = None
    
    def log_audio_stats(self, direction: str, input_data: bytes, output_data: bytes, source: str):
        """Log detailed audio processing statistics with comprehensive analysis"""
        if input_data and output_data:
            input_samples = len(input_data) // 2  # 16-bit samples
            output_samples = len(output_data) // 2
            
            # Calculate audio characteristics
            input_duration_ms = (input_samples / (VICIAL_SAMPLE_RATE if direction == "upsample" else LOCAL_MODEL_SAMPLE_RATE)) * 1000
            output_duration_ms = (output_samples / (LOCAL_MODEL_SAMPLE_RATE if direction == "upsample" else VICIAL_SAMPLE_RATE)) * 1000
            
            # ENHANCED AUDIO ANALYSIS: Analyze input audio samples
            input_array = np.frombuffer(input_data, dtype=np.int16)
            input_max_amp = np.max(np.abs(input_array))
            input_mean_amp = np.mean(np.abs(input_array))
            input_rms = np.sqrt(np.mean(input_array.astype(np.float32)**2))
            input_nonzero_samples = np.count_nonzero(input_array)
            input_zero_percentage = (1.0 - (input_nonzero_samples / len(input_array))) * 100
            
            # ENHANCED AUDIO ANALYSIS: Analyze output audio samples
            output_array = np.frombuffer(output_data, dtype=np.int16)
            output_max_amp = np.max(np.abs(output_array))
            output_mean_amp = np.mean(np.abs(output_array))
            output_rms = np.sqrt(np.mean(output_array.astype(np.float32)**2))
            
            # CRITICAL DEBUG: Log first 20 samples for pattern analysis
            input_first_20 = input_array[:20].tolist()
            output_first_20 = output_array[:20].tolist()
            
            # Determine audio quality indicators
            input_quality = "SILENCE" if input_max_amp < 50 else "LOW" if input_max_amp < 500 else "NORMAL" if input_max_amp < 5000 else "HIGH"
            output_quality = "SILENCE" if output_max_amp < 50 else "LOW" if output_max_amp < 500 else "NORMAL" if output_max_amp < 5000 else "HIGH"
            
            # CRITICAL FIX: Check for timing consistency
            timing_ratio = output_duration_ms / input_duration_ms if input_duration_ms > 0 else 0
            timing_ok = 0.95 <= timing_ratio <= 1.05  # Within 5% tolerance
            
            logger.info(f"🎵 Audio {direction.upper()} | {source}")
            logger.info(f"   📊 Input:  {len(input_data)}B ({input_samples} samples, {input_duration_ms:.1f}ms)")
            logger.info(f"   📊 Output: {len(output_data)}B ({output_samples} samples, {output_duration_ms:.1f}ms)")
            logger.info(f"   ⏱️  Timing: {timing_ratio:.3f}x {'✅' if timing_ok else '❌'} (should be ~1.0x)")
            logger.info(f"   🔊 Input Analysis:  max={input_max_amp}, mean={input_mean_amp:.1f}, rms={input_rms:.1f}, zeros={input_zero_percentage:.1f}% [{input_quality}]")
            logger.info(f"   🔊 Output Analysis: max={output_max_amp}, mean={output_mean_amp:.1f}, rms={output_rms:.1f} [{output_quality}]")
            logger.info(f"   🔍 Input First 20:  {input_first_20}")
            logger.info(f"   🔍 Output First 20: {output_first_20}")
            
            # CRITICAL WARNING: Alert if timing is off (speed issue)
            if not timing_ok:
                logger.error(f"🚨 CRITICAL: Audio timing issue detected! Ratio: {timing_ratio:.3f}x - this will cause speed problems!")
            
            # WARNING: Alert if input appears to be silence or noise
            if input_max_amp < 100:
                logger.warning(f"⚠️  WARNING: Input audio appears to be SILENCE or NOISE (max_amp={input_max_amp})")
            elif input_zero_percentage > 80:
                logger.warning(f"⚠️  WARNING: Input audio is mostly ZEROS ({input_zero_percentage:.1f}% zeros)")
            elif input_max_amp > 0 and input_max_amp < 500:
                logger.warning(f"⚠️  WARNING: Input audio is very LOW LEVEL (max_amp={input_max_amp}) - may not be speech")
    
    def upsample_8k_to_16k(self, audio_data: bytes) -> bytes:
        """
        Convert 8kHz audio to 16kHz using optimized NumPy linear interpolation
        
        Args:
            audio_data: Raw 16-bit PCM audio data at 8kHz
            
        Returns:
            Raw 16-bit PCM audio data at 16kHz
        """
        try:
            if not audio_data or len(audio_data) < 2:
                logger.debug("Empty or insufficient audio data for upsampling")
                return b""
            
            # Ensure even number of bytes (16-bit samples)
            if len(audio_data) % 2 != 0:
                audio_data = audio_data + b'\x00'
                logger.debug("Padded odd-length audio data")
            
            # LATENCY OPTIMIZATION: Use NumPy for faster array operations
            src = np.frombuffer(audio_data, dtype=np.int16)
            
            if len(src) < 2:
                # Handle single sample case
                if len(src) == 1:
                    out = np.full(UPSAMPLE_FACTOR, src[0], dtype=np.int16)
                    return out.tobytes()
                else:
                    return b""
            
            # Perform optimized linear interpolation upsampling with NumPy
            # Create output array with pre-allocated size
            out_size = (len(src) - 1) * UPSAMPLE_FACTOR + 1
            out = np.empty(out_size, dtype=np.int16)
            
            # Vectorized operation: place original samples at correct positions
            out[::UPSAMPLE_FACTOR] = src
            
            # Vectorized interpolation for intermediate samples
            for j in range(1, UPSAMPLE_FACTOR):
                # Calculate interpolation weights
                weight = j / UPSAMPLE_FACTOR
                # Interpolate between consecutive samples
                out[j::UPSAMPLE_FACTOR] = (1 - weight) * src[:-1] + weight * src[1:]
            
            # Update statistics
            self.stats['frames_upsampled'] += 1
            self.stats['total_input_bytes'] += len(audio_data)
            output_data = out.tobytes()
            self.stats['total_output_bytes'] += len(output_data)
            
            self.log_audio_stats("upsample", audio_data, output_data, "8kHz->16kHz")
            return output_data
            
        except Exception as e:
            logger.error(f"Error in upsample_8k_to_16k: {e}")
            self.stats['errors'] += 1
            return b""
    
    def downsample_16k_to_8k(self, audio_data: bytes) -> bytes:
        """
        Convert 16kHz audio to 8kHz using proper decimation (every 2nd sample)
        This ensures correct timing and prevents speed issues
        
        Args:
            audio_data: Raw 16-bit PCM audio data at 16kHz
            
        Returns:
            Raw 16-bit PCM audio data at 8kHz
        """
        try:
            if not audio_data or len(audio_data) < 2:
                logger.debug("Empty or insufficient audio data for downsampling")
                return b""
            
            # Ensure even number of bytes (16-bit samples)
            if len(audio_data) % 2 != 0:
                audio_data = audio_data + b'\x00'
                logger.debug("Padded odd-length audio data")
            
            # LATENCY OPTIMIZATION: Use NumPy for faster array operations
            src = np.frombuffer(audio_data, dtype=np.int16)
            
            if len(src) < DOWNSAMPLE_FACTOR:
                logger.debug(f"Insufficient samples for downsampling: {len(src)} < {DOWNSAMPLE_FACTOR}")
                return b""
            
            # CRITICAL FIX: Use proper decimation (every 2nd sample) instead of averaging
            # This preserves the original timing and prevents speed issues
            out = src[::DOWNSAMPLE_FACTOR].astype(np.int16)
            
            # Update statistics
            self.stats['frames_downsampled'] += 1
            self.stats['total_input_bytes'] += len(audio_data)
            output_data = out.tobytes()
            self.stats['total_output_bytes'] += len(output_data)
            
            self.log_audio_stats("downsample", audio_data, output_data, "16kHz->8kHz")
            return output_data
            
        except Exception as e:
            logger.error(f"Error in downsample_16k_to_8k: {e}")
            self.stats['errors'] += 1
            return b""
    
    def get_stats(self) -> Dict[str, Any]:
        """Return current processing statistics"""
        return self.stats.copy()
    
    async def save_debug_audio(self, call_id: str, audio_data: bytes, audio_type: str, sample_rate: int = 8000):
        """Save debug audio files for manual verification"""
        try:
            if not SAVE_DEBUG_AUDIO or not audio_data:
                return
            
            # Create directory if it doesn't exist
            import os
            os.makedirs(DEBUG_AUDIO_DIR, exist_ok=True)
            
            # Create filename with timestamp and call info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            safe_call_id = call_id.replace(":", "_").replace("/", "_")
            
            # Save as WAV file with proper headers
            filename = f"debug_{audio_type}_{safe_call_id}_{timestamp}.wav"
            filepath = os.path.join(DEBUG_AUDIO_DIR, filename)
            
            # Create WAV file with proper headers
            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setnchannels(1)      # Mono
                wav_file.setsampwidth(2)      # 16-bit
                wav_file.setframerate(sample_rate)   # Sample rate
                wav_file.writeframes(audio_data)
            
            # Log audio analysis
            samples = len(audio_data) // 2
            duration_ms = (samples / sample_rate) * 1000
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            max_amp = np.max(np.abs(audio_array))
            mean_amp = np.mean(np.abs(audio_array))
            nonzero_samples = np.count_nonzero(audio_array)
            zero_percentage = (1.0 - (nonzero_samples / len(audio_array))) * 100
            
            logger.info(f"💾 DEBUG AUDIO SAVED: {filename}")
            logger.info(f"   📊 {len(audio_data)}B, {samples} samples, {duration_ms:.1f}ms @ {sample_rate}Hz")
            logger.info(f"   🔊 max_amp={max_amp}, mean_amp={mean_amp:.1f}, zeros={zero_percentage:.1f}%")
            logger.info(f"   📁 Path: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving debug audio: {e}")

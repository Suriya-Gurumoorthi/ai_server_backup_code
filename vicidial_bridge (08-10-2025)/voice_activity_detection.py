"""
Voice Activity Detection module for Vicidial Bridge

Implements enhanced energy-based Voice Activity Detection (VAD) with filtering to prevent
mic taps and background noise from being mistakenly detected as speech.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, List

try:
    from .config import VAD_DEBUG_LOGGING
except ImportError:
    from config import VAD_DEBUG_LOGGING

# Try to import scipy for advanced filtering, fallback to basic implementation if not available
try:
    import scipy.signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available - using basic VAD without advanced filtering")

logger = logging.getLogger(__name__)

class VoiceActivityDetector:
    """
    Enhanced Voice Activity Detection (VAD) to filter out silence, low-level audio,
    mic taps, and background noise. Implements energy-based VAD with advanced filtering
    and debounce logic to prevent false positives from transient noises.
    """
    
    def __init__(self, 
                 energy_threshold: int = 500,
                 silence_duration_ms: int = 500,
                 min_speech_duration_ms: int = 200,
                 sample_rate: int = 8000,
                 high_pass_cutoff: float = 200.0,
                 min_consecutive_frames: int = 3,
                 spectral_flatness_threshold: float = 0.8):
        self.energy_threshold = energy_threshold
        self.silence_duration_ms = silence_duration_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.sample_rate = sample_rate
        self.high_pass_cutoff = high_pass_cutoff
        self.min_consecutive_frames = min_consecutive_frames
        self.spectral_flatness_threshold = spectral_flatness_threshold
        
        # Convert durations to samples
        self.silence_samples = int(sample_rate * silence_duration_ms / 1000)
        self.min_speech_samples = int(sample_rate * min_speech_duration_ms / 1000)
        
        # State tracking
        self.audio_buffer = bytearray()
        self.speech_start_sample = 0
        self.silence_start_sample = 0
        self.in_speech = False
        self.total_samples_processed = 0
        
        # Enhanced VAD state tracking
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0
        self.frame_history: List[bool] = []  # Track recent frame decisions
        self.max_history_frames = max(min_consecutive_frames * 2, 10)
        
        # Initialize high-pass filter if scipy is available
        self.filter_coeffs = None
        if SCIPY_AVAILABLE and high_pass_cutoff > 0:
            try:
                # Design Butterworth high-pass filter
                nyquist = sample_rate / 2
                normalized_cutoff = high_pass_cutoff / nyquist
                if normalized_cutoff < 1.0:  # Valid cutoff frequency
                    self.filter_coeffs = scipy.signal.butter(1, normalized_cutoff, btype='high', analog=False)
                    logger.info(f"VAD: High-pass filter initialized (cutoff={high_pass_cutoff}Hz)")
                else:
                    logger.warning(f"VAD: Invalid cutoff frequency {high_pass_cutoff}Hz for sample rate {sample_rate}Hz")
            except Exception as e:
                logger.warning(f"VAD: Failed to initialize high-pass filter: {e}")
                self.filter_coeffs = None
        
        logger.info(f"VAD initialized: energy_threshold={energy_threshold}, "
                   f"silence_duration={silence_duration_ms}ms, "
                   f"min_speech_duration={min_speech_duration_ms}ms, "
                   f"high_pass_cutoff={high_pass_cutoff}Hz, "
                   f"min_consecutive_frames={min_consecutive_frames}")
    
    def high_pass_filter(self, audio_array: np.ndarray) -> np.ndarray:
        """
        Apply high-pass filter to suppress low-frequency taps and rumbles
        
        Args:
            audio_array: Input audio samples
            
        Returns:
            Filtered audio array
        """
        if self.filter_coeffs is None or len(audio_array) < 4:
            return audio_array
        
        try:
            # Apply the filter
            b, a = self.filter_coeffs
            filtered = scipy.signal.lfilter(b, a, audio_array.astype(np.float32))
            return filtered.astype(np.int16)
        except Exception as e:
            logger.debug(f"VAD: High-pass filter failed: {e}")
            return audio_array
    
    def calculate_spectral_flatness(self, audio_array: np.ndarray) -> float:
        """
        Calculate spectral flatness to distinguish speech from noise/taps
        
        Args:
            audio_array: Input audio samples
            
        Returns:
            Spectral flatness value (0-1, higher = more noise-like)
        """
        if len(audio_array) < 64:  # Need minimum samples for FFT
            return 0.5
        
        try:
            # Compute FFT
            fft = np.fft.fft(audio_array.astype(np.float32))
            magnitude = np.abs(fft[:len(fft)//2])  # Take positive frequencies only
            
            # Avoid log(0) by adding small epsilon
            eps = 1e-10
            magnitude = np.maximum(magnitude, eps)
            
            # Calculate geometric and arithmetic means
            geometric_mean = np.exp(np.mean(np.log(magnitude)))
            arithmetic_mean = np.mean(magnitude)
            
            # Spectral flatness
            if arithmetic_mean > 0:
                flatness = geometric_mean / arithmetic_mean
                return float(flatness)
            else:
                return 0.5
        except Exception as e:
            logger.debug(f"VAD: Spectral flatness calculation failed: {e}")
            return 0.5
    
    def is_speech(self, audio_data: bytes) -> bool:
        """
        Enhanced speech detection with filtering to prevent mic taps and noise from triggering
        
        Args:
            audio_data: Raw 16-bit PCM audio data
            
        Returns:
            True if audio appears to contain speech, False otherwise
        """
        if not audio_data or len(audio_data) < 2:
            return False
        
        # Convert to numpy array for analysis
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Apply high-pass filter to suppress low-frequency taps and rumbles
        filtered_audio = self.high_pass_filter(audio_array)
        
        # Calculate RMS energy on filtered audio
        rms_energy = np.sqrt(np.mean(filtered_audio.astype(np.float32) ** 2))
        
        # Calculate peak amplitude on filtered audio
        peak_amplitude = np.max(np.abs(filtered_audio))
        
        # Calculate zero-crossing rate (speech has more zero crossings than silence)
        zero_crossings = np.sum(np.diff(np.sign(filtered_audio)) != 0)
        zcr = zero_crossings / len(filtered_audio) if len(filtered_audio) > 0 else 0
        
        # Calculate spectral flatness to distinguish speech from noise/taps
        spectral_flatness = self.calculate_spectral_flatness(filtered_audio)
        
        # Enhanced speech detection criteria
        has_energy = rms_energy > self.energy_threshold or peak_amplitude > self.energy_threshold
        has_activity = zcr > 0.01  # At least 1% zero crossings
        is_not_noise = spectral_flatness < self.spectral_flatness_threshold  # Not too flat/noisy
        
        # Basic speech criteria
        basic_speech = has_energy and has_activity and is_not_noise
        
        # Add to frame history for consecutive frame validation
        self.frame_history.append(basic_speech)
        if len(self.frame_history) > self.max_history_frames:
            self.frame_history.pop(0)
        
        # Count consecutive speech frames
        if basic_speech:
            self.consecutive_speech_frames += 1
            self.consecutive_silence_frames = 0
        else:
            self.consecutive_speech_frames = 0
            self.consecutive_silence_frames += 1
        
        # Require minimum consecutive frames for speech detection (debounce logic)
        has_consecutive_speech = self.consecutive_speech_frames >= self.min_consecutive_frames
        
        # Final decision: must meet basic criteria AND have consecutive frames
        is_speech = basic_speech and has_consecutive_speech
        
        # Log detailed analysis for debugging
        if VAD_DEBUG_LOGGING:
            logger.debug(f"VAD Analysis: rms={rms_energy:.1f}, peak={peak_amplitude}, "
                        f"zcr={zcr:.3f}, flatness={spectral_flatness:.3f}, "
                        f"consecutive={self.consecutive_speech_frames}, "
                        f"basic={basic_speech}, final={is_speech}")
        
        return is_speech
    
    def is_speech_for_barge_in(self, audio_data: bytes) -> bool:
        """
        Fast speech detection for real-time barge-in with lower consecutive frame requirement
        
        Args:
            audio_data: Raw 16-bit PCM audio data
            
        Returns:
            True if audio appears to contain speech for barge-in detection, False otherwise
        """
        if not audio_data or len(audio_data) < 2:
            return False
        
        # Convert to numpy array for analysis
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Apply high-pass filter to suppress low-frequency taps and rumbles
        filtered_audio = self.high_pass_filter(audio_array)
        
        # Calculate RMS energy on filtered audio
        rms_energy = np.sqrt(np.mean(filtered_audio.astype(np.float32) ** 2))
        
        # Calculate peak amplitude on filtered audio
        peak_amplitude = np.max(np.abs(filtered_audio))
        
        # Calculate zero-crossing rate (speech has more zero crossings than silence)
        zero_crossings = np.sum(np.diff(np.sign(filtered_audio)) != 0)
        zcr = zero_crossings / len(filtered_audio) if len(filtered_audio) > 0 else 0
        
        # Calculate spectral flatness to distinguish speech from noise/taps
        spectral_flatness = self.calculate_spectral_flatness(filtered_audio)
        
        # Enhanced speech detection criteria
        has_energy = rms_energy > self.energy_threshold or peak_amplitude > self.energy_threshold
        has_activity = zcr > 0.01  # At least 1% zero crossings
        is_not_noise = spectral_flatness < self.spectral_flatness_threshold  # Not too flat/noisy
        
        # Basic speech criteria
        basic_speech = has_energy and has_activity and is_not_noise
        
        # For barge-in, use a separate consecutive frame counter with lower threshold
        if not hasattr(self, 'barge_in_consecutive_frames'):
            self.barge_in_consecutive_frames = 0
        
        if basic_speech:
            self.barge_in_consecutive_frames += 1
        else:
            self.barge_in_consecutive_frames = 0
        
        # Use lower consecutive frame requirement for barge-in (more responsive)
        # This will be set by the bridge when initializing VAD
        barge_in_threshold = getattr(self, 'barge_in_consecutive_frames_threshold', 2)
        has_consecutive_speech = self.barge_in_consecutive_frames >= barge_in_threshold
        
        # Final decision: must meet basic criteria AND have consecutive frames
        is_speech = basic_speech and has_consecutive_speech
        
        # Log detailed analysis for debugging
        if VAD_DEBUG_LOGGING:
            logger.debug(f"VAD Barge-in Analysis: rms={rms_energy:.1f}, peak={peak_amplitude}, "
                        f"zcr={zcr:.3f}, flatness={spectral_flatness:.3f}, "
                        f"consecutive={self.barge_in_consecutive_frames}, "
                        f"threshold={barge_in_threshold}, basic={basic_speech}, final={is_speech}")
        
        return is_speech
    
    def process_audio_chunk(self, audio_data: bytes) -> Optional[bytes]:
        """
        Process audio chunk and return complete speech segment when detected
        
        Args:
            audio_data: Raw 16-bit PCM audio data
            
        Returns:
            Complete speech segment as bytes, or None if no complete speech detected
        """
        if not audio_data:
            return None
        
        # Add to buffer
        self.audio_buffer.extend(audio_data)
        chunk_samples = len(audio_data) // 2  # 16-bit samples
        
        # Check if current chunk contains speech
        chunk_is_speech = self.is_speech(audio_data)
        
        if chunk_is_speech:
            if not self.in_speech:
                # Start of speech detected
                self.in_speech = True
                self.speech_start_sample = self.total_samples_processed
                self.silence_start_sample = 0
                logger.debug("VAD: Speech started")
            else:
                # Continue speech - reset silence counter
                self.silence_start_sample = 0
        else:
            if self.in_speech:
                # In speech but current chunk is silence
                if self.silence_start_sample == 0:
                    self.silence_start_sample = self.total_samples_processed
                
                # Check if we've had enough silence to end speech
                silence_duration_samples = self.total_samples_processed - self.silence_start_sample
                if silence_duration_samples >= self.silence_samples:
                    # End of speech detected
                    speech_duration_samples = self.silence_start_sample - self.speech_start_sample
                    speech_duration_ms = (speech_duration_samples / self.sample_rate) * 1000
                    
                    if speech_duration_ms >= self.min_speech_duration_ms:
                        # Extract speech segment from buffer
                        speech_bytes = speech_duration_samples * 2  # 16-bit samples
                        
                        if len(self.audio_buffer) >= speech_bytes:
                            speech_segment = bytes(self.audio_buffer[:speech_bytes])
                            # Remove the speech segment from buffer
                            self.audio_buffer = self.audio_buffer[speech_bytes:]
                            
                            logger.info(f"VAD: Complete speech segment detected: "
                                      f"{len(speech_segment)} bytes, {speech_duration_ms:.1f}ms")
                            
                            # Reset state
                            self.in_speech = False
                            self.speech_start_sample = 0
                            self.silence_start_sample = 0
                            
                            return speech_segment
                        else:
                            logger.warning("VAD: Buffer underrun - speech segment too short")
                    else:
                        logger.debug(f"VAD: Speech too short ({speech_duration_ms:.1f}ms) - discarding")
                    
                    # Reset state
                    self.in_speech = False
                    self.speech_start_sample = 0
                    self.silence_start_sample = 0
            else:
                # Not in speech, discard silence
                self.audio_buffer.clear()
        
        # Update total samples processed
        self.total_samples_processed += chunk_samples
        
        # Check for buffer overflow (prevent memory issues)
        max_buffer_size = self.sample_rate * 10 * 2  # 10 seconds max
        if len(self.audio_buffer) > max_buffer_size:
            logger.warning("VAD: Buffer overflow - clearing buffer")
            self.audio_buffer.clear()
            self.in_speech = False
            self.speech_start_sample = 0
            self.silence_start_sample = 0
        
        return None
    
    def flush_remaining_speech(self) -> Optional[bytes]:
        """
        Flush any remaining speech from buffer (call at end of audio stream)
        
        Returns:
            Remaining speech segment or None
        """
        if not self.audio_buffer or not self.in_speech:
            return None
        
        speech_duration_samples = self.total_samples_processed - self.speech_start_sample
        speech_duration_ms = (speech_duration_samples / self.sample_rate) * 1000
        
        if speech_duration_ms >= self.min_speech_duration_ms:
            speech_segment = bytes(self.audio_buffer)
            logger.info(f"VAD: Flushed remaining speech: {len(speech_segment)} bytes, "
                       f"{speech_duration_ms:.1f}ms")
            
            # Reset state
            self.audio_buffer.clear()
            self.in_speech = False
            self.speech_start_sample = 0
            self.silence_start_sample = 0
            
            return speech_segment
        else:
            logger.debug(f"VAD: Flushed speech too short ({speech_duration_ms:.1f}ms) - discarding")
            self.audio_buffer.clear()
            self.in_speech = False
            self.speech_start_sample = 0
            self.silence_start_sample = 0
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get VAD statistics including enhanced parameters"""
        return {
            'buffer_size': len(self.audio_buffer),
            'in_speech': self.in_speech,
            'speech_start_sample': self.speech_start_sample,
            'silence_start_sample': self.silence_start_sample,
            'total_samples_processed': self.total_samples_processed,
            'energy_threshold': self.energy_threshold,
            'silence_duration_ms': self.silence_duration_ms,
            'min_speech_duration_ms': self.min_speech_duration_ms,
            'high_pass_cutoff': self.high_pass_cutoff,
            'min_consecutive_frames': self.min_consecutive_frames,
            'spectral_flatness_threshold': self.spectral_flatness_threshold,
            'consecutive_speech_frames': self.consecutive_speech_frames,
            'consecutive_silence_frames': self.consecutive_silence_frames,
            'frame_history_length': len(self.frame_history),
            'filter_enabled': self.filter_coeffs is not None,
            'scipy_available': SCIPY_AVAILABLE
        }

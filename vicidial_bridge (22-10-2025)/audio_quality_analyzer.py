"""
Audio Quality Analyzer for Vicidial Bridge

Analyzes audio quality to determine if it contains actual speech or just background noise.
Prevents Whisper from hallucinating words when given silence or noise.
"""

import logging
import numpy as np
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

class AudioQualityAnalyzer:
    """
    Analyzes audio quality to determine if it contains speech or just noise.
    Prevents false transcriptions from background noise, silence, or low-quality audio.
    """
    
    def __init__(self, 
                 min_energy_threshold: float = 100.0,
                 min_peak_amplitude: int = 200,
                 min_zero_crossing_rate: float = 0.01,
                 max_spectral_flatness: float = 0.8,
                 min_duration_ms: int = 100):
        """
        Initialize audio quality analyzer with thresholds for speech detection.
        
        Args:
            min_energy_threshold: Minimum RMS energy for speech detection
            min_peak_amplitude: Minimum peak amplitude for speech detection
            min_zero_crossing_rate: Minimum zero crossing rate (speech has more crossings)
            max_spectral_flatness: Maximum spectral flatness (speech is less flat than noise)
            min_duration_ms: Minimum audio duration to consider for analysis
        """
        self.min_energy_threshold = min_energy_threshold
        self.min_peak_amplitude = min_peak_amplitude
        self.min_zero_crossing_rate = min_zero_crossing_rate
        self.max_spectral_flatness = max_spectral_flatness
        self.min_duration_ms = min_duration_ms
        
        logger.info(f"AudioQualityAnalyzer initialized: energy_threshold={min_energy_threshold}, "
                   f"peak_amplitude={min_peak_amplitude}, zcr={min_zero_crossing_rate}, "
                   f"spectral_flatness={max_spectral_flatness}, min_duration={min_duration_ms}ms")
    
    def analyze_audio_quality(self, audio_data: bytes, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Analyze audio quality and determine if it contains speech or just noise.
        
        Args:
            audio_data: Raw audio data bytes
            sample_rate: Sample rate of the audio
            
        Returns:
            Dictionary containing analysis results and quality assessment
        """
        if not audio_data or len(audio_data) < 2:
            return {
                'is_speech': False,
                'is_high_quality': False,
                'should_transcribe': False,
                'reason': 'empty_audio',
                'analysis': {}
            }
        
        # Convert to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Calculate duration
        duration_ms = (len(audio_array) / sample_rate) * 1000
        
        # Check minimum duration
        if duration_ms < self.min_duration_ms:
            return {
                'is_speech': False,
                'is_high_quality': False,
                'should_transcribe': False,
                'reason': 'too_short',
                'analysis': {'duration_ms': duration_ms, 'min_required': self.min_duration_ms}
            }
        
        # Calculate audio characteristics
        analysis = self._calculate_audio_characteristics(audio_array)
        
        # Determine if audio contains speech
        is_speech = self._is_speech_like(analysis)
        
        # Determine if audio is high quality enough for transcription
        is_high_quality = self._is_high_quality_audio(analysis)
        
        # Final decision: only transcribe if it's speech-like AND high quality
        should_transcribe = is_speech and is_high_quality
        
        reason = self._get_rejection_reason(analysis, is_speech, is_high_quality)
        
        return {
            'is_speech': is_speech,
            'is_high_quality': is_high_quality,
            'should_transcribe': should_transcribe,
            'reason': reason,
            'analysis': analysis
        }
    
    def _calculate_audio_characteristics(self, audio_array: np.ndarray) -> Dict[str, float]:
        """Calculate various audio characteristics for quality assessment."""
        # RMS energy
        rms_energy = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
        
        # Peak amplitude
        peak_amplitude = np.max(np.abs(audio_array))
        
        # Zero crossing rate (speech has more zero crossings than silence)
        zero_crossings = np.sum(np.diff(np.sign(audio_array)) != 0)
        zcr = zero_crossings / len(audio_array) if len(audio_array) > 0 else 0
        
        # Spectral flatness (speech is less flat than noise)
        spectral_flatness = self._calculate_spectral_flatness(audio_array)
        
        # Dynamic range (difference between max and min)
        dynamic_range = np.max(audio_array) - np.min(audio_array)
        
        # Signal-to-noise ratio estimation
        signal_power = np.mean(audio_array.astype(np.float32) ** 2)
        noise_floor = np.percentile(np.abs(audio_array), 10)  # Estimate noise floor
        snr_estimate = 10 * np.log10(signal_power / (noise_floor ** 2 + 1e-10))
        
        return {
            'rms_energy': float(rms_energy),
            'peak_amplitude': float(peak_amplitude),
            'zero_crossing_rate': float(zcr),
            'spectral_flatness': float(spectral_flatness),
            'dynamic_range': float(dynamic_range),
            'snr_estimate': float(snr_estimate)
        }
    
    def _calculate_spectral_flatness(self, audio_array: np.ndarray) -> float:
        """Calculate spectral flatness to distinguish speech from noise."""
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
            logger.debug(f"Spectral flatness calculation failed: {e}")
            return 0.5
    
    def _is_speech_like(self, analysis: Dict[str, float]) -> bool:
        """Determine if audio characteristics suggest speech."""
        # Check energy threshold
        has_energy = (analysis['rms_energy'] > self.min_energy_threshold or 
                     analysis['peak_amplitude'] > self.min_peak_amplitude)
        
        # Check zero crossing rate (speech has more crossings)
        has_activity = analysis['zero_crossing_rate'] > self.min_zero_crossing_rate
        
        # Check spectral flatness (speech is less flat than noise)
        is_not_noise = analysis['spectral_flatness'] < self.max_spectral_flatness
        
        # Check dynamic range (speech has more variation)
        has_dynamic_range = analysis['dynamic_range'] > 100
        
        return has_energy and has_activity and is_not_noise and has_dynamic_range
    
    def _is_high_quality_audio(self, analysis: Dict[str, float]) -> bool:
        """Determine if audio is high quality enough for reliable transcription."""
        # Check SNR estimate
        good_snr = analysis['snr_estimate'] > 10  # At least 10dB SNR
        
        # Check that we have reasonable signal strength
        adequate_signal = analysis['rms_energy'] > 50
        
        # Check that it's not too noisy (low spectral flatness)
        not_too_noisy = analysis['spectral_flatness'] < 0.9
        
        return good_snr and adequate_signal and not_too_noisy
    
    def _get_rejection_reason(self, analysis: Dict[str, float], is_speech: bool, is_high_quality: bool) -> str:
        """Get human-readable reason for why audio was rejected."""
        if not is_speech:
            if analysis['rms_energy'] < self.min_energy_threshold:
                return f"low_energy_{analysis['rms_energy']:.1f}"
            elif analysis['zero_crossing_rate'] < self.min_zero_crossing_rate:
                return f"low_activity_{analysis['zero_crossing_rate']:.3f}"
            elif analysis['spectral_flatness'] > self.max_spectral_flatness:
                return f"too_noisy_{analysis['spectral_flatness']:.3f}"
            else:
                return "not_speech_like"
        
        if not is_high_quality:
            if analysis['snr_estimate'] < 10:
                return f"poor_snr_{analysis['snr_estimate']:.1f}dB"
            elif analysis['rms_energy'] < 50:
                return f"weak_signal_{analysis['rms_energy']:.1f}"
            else:
                return "low_quality"
        
        return "passed"
    
    def should_transcribe_audio(self, audio_data: bytes, sample_rate: int = 16000) -> Tuple[bool, str]:
        """
        Quick check to determine if audio should be transcribed.
        
        Args:
            audio_data: Raw audio data bytes
            sample_rate: Sample rate of the audio
            
        Returns:
            Tuple of (should_transcribe, reason)
        """
        quality_result = self.analyze_audio_quality(audio_data, sample_rate)
        return quality_result['should_transcribe'], quality_result['reason']
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer configuration and statistics."""
        return {
            'min_energy_threshold': self.min_energy_threshold,
            'min_peak_amplitude': self.min_peak_amplitude,
            'min_zero_crossing_rate': self.min_zero_crossing_rate,
            'max_spectral_flatness': self.max_spectral_flatness,
            'min_duration_ms': self.min_duration_ms
        }

"""
VAD Manager Module for UltraVAD Integration
Handles real-time voice activity detection using Silero VAD v5 for interruption detection.
"""

import logging
import asyncio
import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

try:
    import torch
    from torch import nn
    import transformers
    from transformers import AutoModel, AutoTokenizer
except ImportError as e:
    print(f"Warning: PyTorch/Transformers dependencies not available: {e}")
    torch = None
    transformers = None
    AutoModel = None
    AutoTokenizer = None
    nn = None

from config import (
    VAD_ENABLED, VAD_MODEL_NAME, VAD_THRESHOLD, 
    VAD_MIN_SPEECH_DURATION_MS, VAD_SPEECH_PAD_MS, 
    VAD_FRAME_SIZE_MS, VAD_SAMPLE_RATE
)


class VADManager:
    """Manages Voice Activity Detection using Silero VAD v5."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.vad_model = None
        self.vad_utils = None
        self.is_loaded = False
        self.frame_size_samples = int(VAD_FRAME_SIZE_MS * VAD_SAMPLE_RATE / 1000)
        self.speech_pad_samples = int(VAD_SPEECH_PAD_MS * VAD_SAMPLE_RATE / 1000)
        self.min_speech_duration_samples = int(VAD_MIN_SPEECH_DURATION_MS * VAD_SAMPLE_RATE / 1000)
        
        if VAD_ENABLED:
            self._load_vad_model()
        else:
            self.logger.info("VAD is disabled via configuration")
    
    def _load_vad_model(self):
        """Load UltraVAD model using transformers."""
        if torch is None or transformers is None:
            self.logger.error("PyTorch or Transformers not available - VAD disabled")
            return
            
        try:
            self.logger.info("Loading UltraVAD model...")
            
            # Load UltraVAD model from Hugging Face
            self.logger.info(f"Loading UltraVAD model: {VAD_MODEL_NAME}")
            
            # Load the UltraVAD model as a pipeline
            # Use CPU for VAD to avoid GPU memory conflicts with main models
            from transformers import pipeline
            
            self.vad_model = pipeline(
                model=VAD_MODEL_NAME,
                trust_remote_code=True,
                device_map="cpu"  # Force CPU usage
            )
            
            # Keep on CPU to avoid memory conflicts
            device = torch.device('cpu')
            self.logger.info(f"UltraVAD pipeline loaded on CPU to avoid GPU memory conflicts")
            
            self.is_loaded = True
            self.logger.info(f"UltraVAD model loaded successfully on {device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load UltraVAD model: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Try fallback to a smaller VAD model
            try:
                self.logger.info("Attempting fallback to smaller VAD model...")
                from transformers import pipeline
                self.vad_model = pipeline(
                    model="facebook/wav2vec2-base",  # Smaller fallback model
                    trust_remote_code=True,
                    device_map="cpu"
                )
                self.is_loaded = True
                self.logger.info("Fallback VAD model loaded successfully on CPU")
            except Exception as fallback_error:
                self.logger.error(f"Fallback VAD model also failed: {fallback_error}")
                self.logger.warning("VAD functionality will be disabled")
                self.is_loaded = False
    
    def is_vad_available(self) -> bool:
        """Check if VAD functionality is available."""
        return VAD_ENABLED and self.is_loaded and self.vad_model is not None
    
    def detect_speech(self, audio_bytes: bytes, conversation_turns: list = None) -> float:
        """
        Detect speech probability in audio bytes using UltraVAD.
        
        Args:
            audio_bytes: Raw 16-bit PCM audio data
            conversation_turns: Optional conversation context for UltraVAD
            
        Returns:
            float: Speech probability (0.0-1.0)
        """
        if not self.is_vad_available():
            return 0.0
        
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Normalize to float32 range [-1, 1]
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Ensure minimum length
            if len(audio_float) < self.frame_size_samples:
                # Pad with zeros if too short
                audio_float = np.pad(audio_float, (0, self.frame_size_samples - len(audio_float)))
            
            # For UltraVAD, we need conversation context with exactly one audio placeholder
            if conversation_turns is None:
                conversation_turns = [
                    {"role": "assistant", "content": "Hello, how can I help you?"},
                    {"role": "user", "content": "I need help with something <|audio|>"}
                ]
            
            # Prepare inputs for UltraVAD
            inputs = {
                "audio": audio_float,
                "turns": conversation_turns,
                "sampling_rate": VAD_SAMPLE_RATE
            }
            
            # Use UltraVAD pipeline if available
            if hasattr(self.vad_model, 'preprocess'):
                # This is a pipeline
                model_inputs = self.vad_model.preprocess(inputs)
                
                # Move to device
                device = next(self.vad_model.model.parameters()).device
                model_inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in model_inputs.items()}
                
                # Forward pass
                with torch.no_grad():
                    output = self.vad_model.model.forward(**model_inputs, return_dict=True)
                
                # Extract end-of-turn probability
                logits = output.logits
                audio_pos = int(model_inputs["audio_token_start_idx"].item() + model_inputs["audio_token_len"].item() - 1)
                token_id = self.vad_model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                audio_logits = logits[0, audio_pos, :]
                audio_probs = torch.softmax(audio_logits.float(), dim=-1)
                eot_prob_audio = audio_probs[token_id].item()
                
                # Convert end-of-turn probability to speech probability
                # Higher EOT probability means less likely to continue speaking
                speech_prob = 1.0 - eot_prob_audio
                
            else:
                # Fallback for wav2vec2 or other models
                audio_tensor = torch.from_numpy(audio_float).unsqueeze(0)
                device = next(self.vad_model.parameters()).device
                audio_tensor = audio_tensor.to(device)
                
                with torch.no_grad():
                    outputs = self.vad_model(audio_tensor)
                    
                    if hasattr(outputs, 'last_hidden_state'):
                        # Use hidden state for VAD (wav2vec2 fallback)
                        hidden_state = outputs.last_hidden_state
                        variance = torch.var(hidden_state).item()
                        speech_prob = min(1.0, max(0.0, variance * 100))
                    else:
                        speech_prob = 0.5  # Default fallback
            
            return speech_prob
            
        except Exception as e:
            self.logger.error(f"Error in speech detection: {e}")
            return 0.0
    
    def detect_speech_with_temporal_consistency(self, audio_chunks: list, 
                                               required_consecutive_frames: int = 3) -> Tuple[bool, float]:
        """
        Detect speech with temporal consistency to reduce false positives.
        
        Args:
            audio_chunks: List of audio byte chunks
            required_consecutive_frames: Number of consecutive speech frames needed
            
        Returns:
            Tuple[bool, float]: (is_speech_detected, average_speech_probability)
        """
        if not self.is_vad_available() or not audio_chunks:
            return False, 0.0
        
        try:
            speech_probs = []
            consecutive_speech_frames = 0
            max_consecutive = 0
            
            for chunk in audio_chunks:
                if len(chunk) == 0:
                    continue
                    
                speech_prob = self.detect_speech(chunk)
                speech_probs.append(speech_prob)
                
                if speech_prob > VAD_THRESHOLD:
                    consecutive_speech_frames += 1
                    max_consecutive = max(max_consecutive, consecutive_speech_frames)
                else:
                    consecutive_speech_frames = 0
            
            # Check if we have enough consecutive speech frames
            is_speech = max_consecutive >= required_consecutive_frames
            avg_prob = np.mean(speech_probs) if speech_probs else 0.0
            
            self.logger.debug(f"VAD temporal analysis: {len(speech_probs)} frames, "
                            f"max_consecutive={max_consecutive}, avg_prob={avg_prob:.3f}, "
                            f"is_speech={is_speech}")
            
            return is_speech, avg_prob
            
        except Exception as e:
            self.logger.error(f"Error in temporal speech detection: {e}")
            return False, 0.0
    
    def get_speech_segments(self, audio_bytes: bytes, 
                           min_speech_duration_ms: Optional[int] = None) -> list:
        """
        Get speech segments from audio using UltraVAD.
        
        Args:
            audio_bytes: Raw audio data
            min_speech_duration_ms: Minimum speech duration in milliseconds
            
        Returns:
            List of speech segments with start/end times
        """
        if not self.is_vad_available():
            return []
        
        if min_speech_duration_ms is None:
            min_speech_duration_ms = VAD_MIN_SPEECH_DURATION_MS
        
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Process audio in chunks to detect speech segments
            chunk_size = int(VAD_FRAME_SIZE_MS * VAD_SAMPLE_RATE / 1000)
            speech_segments = []
            current_segment_start = None
            
            for i in range(0, len(audio_float), chunk_size):
                chunk = audio_float[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    # Pad last chunk if needed
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                
                # Convert chunk to bytes for speech detection
                chunk_bytes = (chunk * 32767).astype(np.int16).tobytes()
                speech_prob = self.detect_speech(chunk_bytes)
                
                if speech_prob > VAD_THRESHOLD:
                    if current_segment_start is None:
                        current_segment_start = i
                else:
                    if current_segment_start is not None:
                        # End of speech segment
                        end_sample = i
                        duration_samples = end_sample - current_segment_start
                        
                        if duration_samples >= int(min_speech_duration_ms * VAD_SAMPLE_RATE / 1000):
                            start_ms = (current_segment_start / VAD_SAMPLE_RATE) * 1000
                            end_ms = (end_sample / VAD_SAMPLE_RATE) * 1000
                            speech_segments.append({
                                'start_ms': start_ms,
                                'end_ms': end_ms,
                                'duration_ms': end_ms - start_ms,
                                'start_sample': current_segment_start,
                                'end_sample': end_sample
                            })
                        current_segment_start = None
            
            # Handle case where speech continues to end of audio
            if current_segment_start is not None:
                end_sample = len(audio_float)
                duration_samples = end_sample - current_segment_start
                
                if duration_samples >= int(min_speech_duration_ms * VAD_SAMPLE_RATE / 1000):
                    start_ms = (current_segment_start / VAD_SAMPLE_RATE) * 1000
                    end_ms = (end_sample / VAD_SAMPLE_RATE) * 1000
                    speech_segments.append({
                        'start_ms': start_ms,
                        'end_ms': end_ms,
                        'duration_ms': end_ms - start_ms,
                        'start_sample': current_segment_start,
                        'end_sample': end_sample
                    })
            
            self.logger.debug(f"Found {len(speech_segments)} speech segments")
            return speech_segments
            
        except Exception as e:
            self.logger.error(f"Error getting speech segments: {e}")
            return []
    
    def is_speech_present(self, audio_bytes: bytes) -> bool:
        """
        Simple check if speech is present in audio.
        
        Args:
            audio_bytes: Raw audio data
            
        Returns:
            bool: True if speech is detected
        """
        if not self.is_vad_available():
            return False
        
        speech_prob = self.detect_speech(audio_bytes)
        return speech_prob > VAD_THRESHOLD
    
    def get_audio_stats(self, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Get audio statistics for debugging.
        
        Args:
            audio_bytes: Raw audio data
            
        Returns:
            Dictionary with audio statistics
        """
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            stats = {
                'duration_ms': (len(audio_array) / VAD_SAMPLE_RATE) * 1000,
                'samples': len(audio_array),
                'rms_energy': float(np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))),
                'max_amplitude': int(np.max(np.abs(audio_array))),
                'speech_probability': self.detect_speech(audio_bytes) if self.is_vad_available() else 0.0
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting audio stats: {e}")
            return {'error': str(e)}
    
    def shutdown(self):
        """Shutdown VAD manager and clean up resources."""
        self.logger.info("Shutting down VAD manager...")
        self.vad_model = None
        self.is_loaded = False


# Global VAD manager instance
vad_manager = VADManager()

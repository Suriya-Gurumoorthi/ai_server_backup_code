"""
UltraVAD model manager for voice activity detection and interruption detection.
Handles loading and inference of the ultraVAD model on CPU.
"""

import logging
import torch
import librosa
import numpy as np
from typing import Optional, Tuple, Dict, Any
import time

try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class UltraVADManager:
    """Manages ultraVAD model for voice activity detection and interruption detection."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.sample_rate = 16000
        self.threshold = 0.1  # Default threshold for end-of-turn detection
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """Load the ultraVAD model on CPU."""
        if not TRANSFORMERS_AVAILABLE:
            self.logger.error("Transformers library not available. Cannot load ultraVAD model.")
            return False
            
        try:
            self.logger.info("Loading ultraVAD model on CPU...")
            
            # Clear any existing GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load the model pipeline on CPU
            self.model = transformers.pipeline(
                model='fixie-ai/ultraVAD', 
                trust_remote_code=True, 
                device="cpu",
                torch_dtype=torch.float32  # Use float32 for better compatibility
            )
            
            # Extract tokenizer from the pipeline
            self.tokenizer = self.model.tokenizer
            
            self.is_loaded = True
            self.logger.info("✅ ultraVAD model loaded successfully on CPU")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load ultraVAD model: {e}")
            self.is_loaded = False
            return False
    
    def is_model_available(self) -> bool:
        """Check if the ultraVAD model is loaded and available."""
        return self.is_loaded and self.model is not None
    
    def detect_interruption(self, audio_bytes: bytes, conversation_turns: list = None) -> Tuple[bool, float]:
        """
        Detect if there's an interruption or disturbance in the audio.
        
        Args:
            audio_bytes: Raw audio bytes
            conversation_turns: List of conversation turns for context
            
        Returns:
            Tuple of (is_interruption, confidence_score)
        """
        if not self.is_model_available():
            self.logger.warning("UltraVAD model not available, skipping interruption detection")
            return False, 0.0
        
        try:
            # Convert audio bytes to numpy array
            audio_array = self._bytes_to_audio_array(audio_bytes)
            if audio_array is None:
                return False, 0.0
            
            # Prepare conversation turns (use default if none provided)
            if conversation_turns is None:
                conversation_turns = [{"role": "assistant", "content": "Hi, how are you?"}]
            
            # Build model inputs
            inputs = {
                "audio": audio_array, 
                "turns": conversation_turns, 
                "sampling_rate": self.sample_rate
            }
            
            # Preprocess inputs
            model_inputs = self.model.preprocess(inputs)
            
            # Move tensors to model device (CPU)
            device = next(self.model.model.parameters()).device
            model_inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in model_inputs.items()}
            
            # Forward pass
            with torch.inference_mode():
                output = self.model.model.forward(**model_inputs, return_dict=True)
            
            # Compute end-of-turn probability
            logits = output.logits  # (1, seq_len, vocab)
            audio_pos = int(
                model_inputs["audio_token_start_idx"].item() +
                model_inputs["audio_token_len"].item() - 1
            )
            
            # Get end-of-turn token probability
            token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if token_id is None or token_id == self.tokenizer.unk_token_id:
                self.logger.warning("End-of-turn token not found in tokenizer")
                return False, 0.0
            
            audio_logits = logits[0, audio_pos, :]
            audio_probs = torch.softmax(audio_logits.float(), dim=-1)
            eot_prob = audio_probs[token_id].item()
            
            # Determine if this is an interruption (high end-of-turn probability)
            is_interruption = eot_prob > self.threshold
            
            self.logger.debug(f"UltraVAD detection - EOT probability: {eot_prob:.6f}, Threshold: {self.threshold}, Is interruption: {is_interruption}")
            
            return is_interruption, eot_prob
            
        except Exception as e:
            self.logger.error(f"Error in ultraVAD interruption detection: {e}")
            return False, 0.0
    
    def _bytes_to_audio_array(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """Convert audio bytes to numpy array for processing."""
        try:
            # Use librosa to load audio from bytes
            import io
            audio_io = io.BytesIO(audio_bytes)
            audio_array, sr = librosa.load(audio_io, sr=self.sample_rate)
            
            # Ensure audio is not empty
            if len(audio_array) == 0:
                self.logger.warning("Empty audio array received")
                return None
                
            return audio_array
            
        except Exception as e:
            self.logger.error(f"Error converting audio bytes to array: {e}")
            return None
    
    def set_threshold(self, threshold: float):
        """Set the threshold for interruption detection."""
        self.threshold = max(0.0, min(1.0, threshold))  # Clamp between 0 and 1
        self.logger.info(f"UltraVAD threshold set to: {self.threshold}")
    
    def get_threshold(self) -> float:
        """Get the current threshold for interruption detection."""
        return self.threshold
    
    def shutdown(self):
        """Shutdown the ultraVAD manager and clean up resources."""
        if self.model is not None:
            self.logger.info("Shutting down ultraVAD manager...")
            # Clear model from memory
            del self.model
            self.model = None
            self.tokenizer = None
            self.is_loaded = False
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("UltraVAD manager shutdown complete")


# Global ultraVAD manager instance
ultravad_manager = UltraVADManager()

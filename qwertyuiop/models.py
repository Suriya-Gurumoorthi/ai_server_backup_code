"""
AI model management module for Ultravox and Piper TTS.
Handles model loading, initialization, and inference operations.
"""

import os
import logging
import asyncio
from typing import Optional, Any, Dict, List
from concurrent.futures import ThreadPoolExecutor

try:
    import transformers
    import torch
    from piper import PiperVoice
    import librosa
    import numpy as np
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    transformers = None
    torch = None
    PiperVoice = None
    librosa = None
    np = None

from config import (
    DEVICE, ULTRAVOX_MODEL, PIPER_MODEL_NAME, PIPER_ONNX_FILE, 
    PIPER_JSON_FILE, MAX_WORKERS, THREAD_NAME_PREFIX
)
from prompt_logger import prompt_logger
from audio_utils import safe_audio_conversion, debug_audio_bytes, is_valid_wav


def sanitize_audio_placeholders(text: str) -> str:
    """Remove all audio placeholders from text to prevent Ultravox pipeline errors"""
    if not text:
        return text
    
    # Remove all variations of audio placeholders
    text = text.replace('<|audio|>', '')
    text = text.replace('<|audio|', '')
    text = text.replace('|audio|>', '')
    text = text.replace('|audio|', '')
    
    # Clean up any extra spaces
    text = ' '.join(text.split())
    
    return text.strip()


def ensure_one_audio_placeholder_last_user(turns: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Removes all audio placeholders from all turns except the last user turn,
    and ensures only one <|audio|> at the end.
    """
    if not turns:
        return turns
    
    # Find the last user turn
    last_user_idx = -1
    for i in range(len(turns) - 1, -1, -1):
        if turns[i].get("role") == "user":
            last_user_idx = i
            break
    
    if last_user_idx == -1:
        # No user turns found, return as is
        return turns
    
    cleaned = []
    for idx, turn in enumerate(turns):
        role = turn.get("role", "")
        content = turn.get("content", "")

        # Remove all existing placeholders from all turns
        content = sanitize_audio_placeholders(content)
        
        if role == "user" and idx == last_user_idx:
            # Only the last user turn gets the audio marker
            # Ensure we don't add multiple placeholders
            if not content.strip().endswith('<|audio|>'):
                content = content + " <|audio|>"
        
        cleaned.append({
            "role": role,
            "content": content
        })
    
    return cleaned


class ModelManager:
    """Manages AI models and their inference operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ultravox_pipeline = None
        self.piper_voice = None
        self.whisper_processor = None
        self.whisper_model = None
        self.executor = ThreadPoolExecutor(
            max_workers=MAX_WORKERS, 
            thread_name_prefix=THREAD_NAME_PREFIX
        )
        self._load_models()
    
    def _load_models(self):
        """Load all required AI models."""
        self._load_ultravox()
        self._load_piper()
        self._load_whisper()
    
    def _load_ultravox(self):
        """Load the Ultravox pipeline."""
        if transformers is None or torch is None:
            self.logger.error("Transformers or PyTorch not available")
            raise RuntimeError("Required dependencies not available")
            
        self.logger.info("Loading Ultravox pipeline...")
        try:
            self.ultravox_pipeline = transformers.pipeline(
                model=ULTRAVOX_MODEL,
                trust_remote_code=True,
                device=DEVICE,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
            )
            self.logger.info(f"Ultravox pipeline loaded successfully on {DEVICE}!")
            if DEVICE == "cuda":
                self.logger.info("Model is using GPU acceleration for faster inference")
        except Exception as e:
            self.logger.error(f"Failed to load Ultravox pipeline: {e}")
            raise
    
    def _load_piper(self):
        """Load the Piper TTS model."""
        if PiperVoice is None:
            self.logger.warning("Piper TTS not available")
            self.piper_voice = None
            return
            
        self.logger.info("Loading Piper TTS...")
        try:
            # Check for required model files
            if self._check_piper_files():
                self.piper_voice = PiperVoice.load(PIPER_MODEL_NAME)
                self.logger.info("Piper TTS loaded successfully!")
            else:
                self.logger.warning("TTS functionality will be disabled")
                self.piper_voice = None
        except Exception as e:
            self.logger.error(f"Failed to load Piper TTS: {e}")
            self.logger.warning("TTS functionality will be disabled")
            self.piper_voice = None
    
    def _load_whisper(self):
        """Load the Whisper model for speech-to-text transcription."""
        if transformers is None or torch is None:
            self.logger.error("Transformers or PyTorch not available for Whisper")
            self.whisper_processor = None
            self.whisper_model = None
            return
            
        self.logger.info("Loading Whisper model...")
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            
            self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
            
            # Move model to device
            self.whisper_model = self.whisper_model.to(DEVICE)
            
            self.logger.info(f"Whisper model loaded successfully on {DEVICE}!")
            if DEVICE == "cuda":
                self.logger.info("Whisper model is using GPU acceleration for faster inference")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            self.logger.warning("Whisper transcription functionality will be disabled")
            self.whisper_processor = None
            self.whisper_model = None
    
    def _check_piper_files(self) -> bool:
        """Check if all required Piper model files exist."""
        required_files = [PIPER_ONNX_FILE, PIPER_JSON_FILE, PIPER_MODEL_NAME]
        missing_files = []
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            self.logger.warning(f"Missing required files: {missing_files}")
            return False
        
        self.logger.info(f"Found ONNX model file: {PIPER_ONNX_FILE}")
        self.logger.info(f"Found JSON config file: {PIPER_JSON_FILE}")
        self.logger.info(f"Found model symlink: {PIPER_MODEL_NAME}")
        return True
    
    async def process_audio(self, audio_data: bytes, conversation_turns: List[Dict[str, str]], connection_id: str = None) -> str:
        """Process audio input using Ultravox pipeline."""
        if not self.ultravox_pipeline:
            raise RuntimeError("Ultravox pipeline not loaded")
        
        try:
            import librosa
            import io
            
            # Debug audio format
            debug_audio_bytes(audio_data, f"Process Audio (connection: {connection_id})")
            
            # Convert audio to proper WAV format if needed
            converted_audio = safe_audio_conversion(audio_data, target_sample_rate=16000)
            
            # Load audio from bytes
            audio_stream = io.BytesIO(converted_audio)
            audio, sr = librosa.load(audio_stream, sr=16000)
            self.logger.info(f"Loaded audio: {len(audio)} samples at {sr}Hz")
            
            if len(audio) == 0:
                return "Error: Audio file contains no data"
            
            # Ensure proper audio placeholder format
            conversation_turns = ensure_one_audio_placeholder_last_user(conversation_turns)
            
            # Ensure system prompt is always first and consistent
            if not conversation_turns or conversation_turns[0].get("role") != "system":
                # Add system prompt if missing
                system_prompt = {
                    "role": "system",
                    "content": """You are Alexa, an HR recruiter from Novel Office calling Business Development Manager applicants. Speak naturally and professionally, as in a real phone call. Keep responses short, 1–2 sentences at a time. Do not use lists, bullets, emojis, stage directions, or overly formal prose; this is a live voice conversation. Always maintain your identity as Alexa and address the candidate appropriately."""
                }
                conversation_turns = [system_prompt] + conversation_turns
            
            # Log the prompt being sent to Ultravox
            if connection_id:
                prompt_logger.log_ultravox_prompt(
                    connection_id=connection_id,
                    audio_data=audio_data,
                    conversation_turns=conversation_turns
                )
            
            # Run inference in thread executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: self.ultravox_pipeline({
                    'audio': audio, 
                    'turns': conversation_turns, 
                    'sampling_rate': sr
                }, max_new_tokens=2000, do_sample=True, temperature=0.7)
            )
            
            self.logger.info("Ultravox inference completed in thread executor")
            response_text = self._extract_response_text(result)
            
            # Log the response
            if connection_id:
                prompt_logger.log_ultravox_prompt(
                    connection_id=connection_id,
                    audio_data=audio_data,
                    conversation_turns=conversation_turns,
                    response=response_text
                )
            
            return response_text
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Ultravox pipeline error: {error_msg}")
            
            # Handle specific audio placeholder errors
            if "too many audio placeholders" in error_msg.lower():
                self.logger.warning("Audio placeholder error detected - attempting to fix conversation turns")
                # Try to fix the conversation turns and retry once
                try:
                    # Clean all audio placeholders and add only one at the end
                    fixed_turns = []
                    for turn in conversation_turns:
                        content = sanitize_audio_placeholders(turn.get("content", ""))
                        fixed_turns.append({
                            "role": turn.get("role", ""),
                            "content": content
                        })
                    
                    # Add audio placeholder only to the last user turn
                    if fixed_turns and fixed_turns[-1].get("role") == "user":
                        fixed_turns[-1]["content"] = fixed_turns[-1]["content"] + " <|audio|>"
                    
                    # Retry with fixed turns
                    self.logger.info("Retrying with fixed conversation turns")
                    result = self.ultravox_pipeline({
                        'audio': audio, 
                        'turns': fixed_turns, 
                        'sampling_rate': sr
                    }, max_new_tokens=2000, do_sample=True, temperature=0.7)
                    
                    response_text = self._extract_response_text(result)
                    self.logger.info("Ultravox inference completed after fixing audio placeholders")
                    return response_text
                    
                except Exception as retry_error:
                    self.logger.error(f"Retry failed: {retry_error}")
                    return "I apologize, but I'm having trouble processing your audio. Could you please try speaking again?"
            
            return f"Error processing audio with Ultravox: {error_msg}"
    
    def _extract_response_text(self, result: Any) -> str:
        """Extract text response from Ultravox pipeline result."""
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and 'generated_text' in result[0]:
                return result[0]['generated_text']
            elif isinstance(result[0], str):
                return result[0]
            else:
                return str(result[0])
        elif isinstance(result, dict):
            if 'generated_text' in result:
                return result['generated_text']
            elif 'text' in result:
                return result['text']
            else:
                return str(result)
        elif isinstance(result, str):
            return result
        else:
            return str(result)
    
    def _is_conversational_response(self, text: str) -> bool:
        """Check if the response is conversational rather than a pure transcription."""
        if not text:
            return False
        
        # Common conversational phrases that indicate AI responses
        conversational_indicators = [
            "I can't", "I cannot", "I'm not able to", "I don't", "I won't",
            "I can help you", "I understand", "I see", "Let me", "I'll",
            "I'm sorry", "I apologize", "I'm afraid", "I'm doing well",
            "Thank you for asking", "I'm ready when you are",
            "I'm not able to transcribe", "I can't provide guidance",
            "I cannot provide information", "I can't give you any information"
        ]
        
        text_lower = text.lower().strip()
        
        # Check if it starts with conversational indicators
        for indicator in conversational_indicators:
            if text_lower.startswith(indicator.lower()):
                return True
        
        # Check if it's a very long response (transcriptions should be relatively short)
        if len(text) > 200:
            return True
            
        # Check if it contains multiple sentences with conversational structure
        sentences = text.split('.')
        if len(sentences) > 3:
            return True
            
        return False
    
    def _extract_transcription_from_response(self, response: str) -> str:
        """Try to extract actual transcription from a conversational response."""
        if not response:
            return response
        
        # Look for quoted text that might be the actual transcription
        import re
        
        # Look for text in quotes
        quoted_matches = re.findall(r'"([^"]*)"', response)
        if quoted_matches:
            # Return the longest quoted text (likely the transcription)
            return max(quoted_matches, key=len)
        
        # Look for text after common transcription indicators
        transcription_indicators = [
            "the audio says:", "the transcription is:", "the text is:",
            "transcribed as:", "says:", "audio contains:"
        ]
        
        text_lower = response.lower()
        for indicator in transcription_indicators:
            if indicator in text_lower:
                # Extract text after the indicator
                parts = response.split(indicator, 1)
                if len(parts) > 1:
                    extracted = parts[1].strip()
                    # Remove any trailing conversational text
                    sentences = extracted.split('.')
                    if sentences:
                        return sentences[0].strip()
        
        # If no clear transcription found, return the original response
        # but truncate if it's too long
        if len(response) > 100:
            return response[:100] + "..."
        
        return response
    
    async def transcribe_audio(self, audio_data: bytes, connection_id: str = None) -> str:
        """Transcribe audio input to text using Whisper model."""
        if not self.whisper_processor or not self.whisper_model:
            raise RuntimeError("Whisper model not loaded")
        
        try:
            import io
            
            # Debug audio format
            debug_audio_bytes(audio_data, f"Transcribe Audio (connection: {connection_id})")
            
            # Convert audio to proper WAV format if needed
            converted_audio = safe_audio_conversion(audio_data, target_sample_rate=16000)
            
            # Load audio from bytes
            audio_stream = io.BytesIO(converted_audio)
            audio, sr = librosa.load(audio_stream, sr=16000)
            self.logger.info(f"Loaded audio for transcription: {len(audio)} samples at {sr}Hz")
            
            if len(audio) == 0:
                return "Error: Audio file contains no data"
            
            # Log the prompt being sent to Whisper
            if connection_id:
                prompt_logger.log_whisper_prompt(
                    connection_id=connection_id,
                    audio_data=audio_data
                )
            
            # Calculate duration and determine if we need chunking
            duration = len(audio) / sr
            chunk_length = 30  # 30 seconds per chunk
            
            # If audio is short, process normally
            if duration <= chunk_length:
                result = await self._transcribe_audio_chunk(audio, sr)
            else:
                # For longer audio, process in chunks
                self.logger.info(f"Processing audio in {chunk_length}-second chunks...")
                chunk_samples = chunk_length * sr
                full_transcription = []
                
                for i in range(0, len(audio), chunk_samples):
                    chunk = audio[i:i + chunk_samples]
                    if len(chunk) < sr:  # Skip very short chunks
                        break
                        
                    self.logger.info(f"Processing chunk {i//chunk_samples + 1}/{(len(audio) + chunk_samples - 1)//chunk_samples}")
                    
                    # Process chunk
                    chunk_transcription = await self._transcribe_audio_chunk(chunk, sr)
                    if chunk_transcription.strip():
                        full_transcription.append(chunk_transcription)
                
                # Combine all chunks
                result = " ".join(full_transcription)
            
            # Log the response
            if connection_id:
                prompt_logger.log_whisper_prompt(
                    connection_id=connection_id,
                    audio_data=audio_data,
                    response=result
                )
            
            self.logger.info(f"Final transcription result: {result[:100]}...")
            return result
            
        except Exception as e:
            self.logger.error(f"Audio transcription error: {e}")
            return f"Error transcribing audio: {str(e)}"
    
    async def _transcribe_audio_chunk(self, audio: np.ndarray, sr: int) -> str:
        """Transcribe a single audio chunk using Whisper."""
        try:
            # Process audio
            input_features = self.whisper_processor(audio, sampling_rate=sr, return_tensors="pt").input_features
            
            # Move input features to the same device as the model
            input_features = input_features.to(DEVICE)
            
            # Generate transcription
            with torch.no_grad():  # Save memory
                predicted_ids = self.whisper_model.generate(input_features, max_length=448)
            
            # Decode to text
            transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
            return transcription[0]
            
        except Exception as e:
            self.logger.error(f"Error transcribing audio chunk: {e}")
            return ""
    
    async def process_text_with_context(self, user_text: str, conversation_turns: List[Dict[str, str]], conversation_history: List[Dict[str, str]]) -> str:
        """Process text input with conversation context using Ultravox pipeline."""
        if not self.ultravox_pipeline:
            raise RuntimeError("Ultravox pipeline not loaded")
        
        try:
            # Combine conversation turns with history
            combined_turns = conversation_turns.copy()
            
            # Add conversation history
            for entry in conversation_history:
                combined_turns.append(entry)
            
            # Add the current user input
            combined_turns.append({"role": "user", "content": user_text})
            
            # Ensure proper audio placeholder format
            combined_turns = ensure_one_audio_placeholder_last_user(combined_turns)
            
            # Ensure system prompt is always first and consistent
            if not combined_turns or combined_turns[0].get("role") != "system":
                # Add system prompt if missing
                system_prompt = {
                    "role": "system",
                    "content": """You are Alexa, an HR recruiter from Novel Office calling Business Development Manager applicants. Speak naturally and professionally, as in a real phone call. Keep responses short, 1–2 sentences at a time. Do not use lists, bullets, emojis, stage directions, or overly formal prose; this is a live voice conversation. Always maintain your identity as Alexa and address the candidate appropriately."""
                }
                combined_turns = [system_prompt] + combined_turns
            
            # Create a simple audio placeholder (silence) since we're processing text
            import numpy as np
            silence_audio = np.zeros(1600, dtype=np.float32)  # 0.1 seconds of silence at 16kHz
            
            # Run inference in thread executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: self.ultravox_pipeline({
                    'audio': silence_audio, 
                    'turns': combined_turns, 
                    'sampling_rate': 16000
                }, max_new_tokens=2000, do_sample=True, temperature=0.7)
            )
            
            self.logger.info("Text processing with context completed")
            response_text = self._extract_response_text(result)
            # Sanitize any audio placeholders from the response
            return sanitize_audio_placeholders(response_text)
            
        except Exception as e:
            self.logger.error(f"Text processing error: {e}")
            return f"Error processing text: {str(e)}"
    
    def is_tts_available(self) -> bool:
        """Check if TTS functionality is available."""
        return self.piper_voice is not None
    
    async def summarize_conversation(self, conversation_history: List[dict], connection_id: str = None) -> str:
        """Run the Ultravox/Llama model to summarize the conversation."""
        if not self.ultravox_pipeline:
            raise RuntimeError("Ultravox pipeline not loaded")

        # Defensive checks
        if not isinstance(conversation_history, list):
            raise ValueError(f"Conversation history must be a list, got {type(conversation_history)}")
        
        if not conversation_history:
            return "No conversation history available for summarization."
        
        for i, entry in enumerate(conversation_history):
            if not isinstance(entry, dict):
                raise ValueError(f"Entry {i} must be a dict, got {type(entry)}: {entry}")
            if "type" not in entry:
                raise ValueError(f"Entry {i} missing 'type' field: {entry}")

        # Format for LLM using the proper pipeline format, sanitizing any audio markers
        turns = []
        for entry in conversation_history:
            if entry["type"] == "user":
                # Remove any <|audio|> from user text
                user_text = sanitize_audio_placeholders(entry['User'])
                turns.append({"role": "user", "content": f"Candidate: {user_text}"})
            elif entry["type"] == "ai":
                ai_text = sanitize_audio_placeholders(entry['AI'])
                turns.append({"role": "assistant", "content": f"Recruiter: {ai_text}"})

        # Add system prompt for summarization (sanitized)
        system_prompt = {
            "role": "system", 
            "content": sanitize_audio_placeholders(
                "You are a CRM assistant. Summarize the following recruiter-candidate interview in 5-7 lines for CRM notes. Mention candidate's background, key details, interviewer remarks, outcome, and candidate's interest/fit. Be concise and professional."
            )
        }
        
        # Add final user prompt requesting summary (sanitized)
        summary_request = {
            "role": "user",
            "content": sanitize_audio_placeholders("Please provide a summary of this interview conversation.")
        }
        
        # Combine all turns
        all_turns = [system_prompt] + turns + [summary_request]
        
        # Ensure proper audio placeholder format for Ultravox pipeline
        all_turns = ensure_one_audio_placeholder_last_user(all_turns)
        
        # Log the prompt being sent for summarization
        if connection_id:
            prompt_logger.log_summarization_prompt(
                connection_id=connection_id,
                conversation_history=conversation_history
            )
        
        # Create silence audio for the pipeline (since we're doing text-only summarization)
        import numpy as np
        silence_audio = np.zeros(1600, dtype=np.float32)  # 0.1 seconds of silence at 16kHz

        # Send to LLM for summarization using proper pipeline format
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            lambda: self.ultravox_pipeline({
                'audio': silence_audio, 
                'turns': all_turns, 
                'sampling_rate': 16000
            }, max_new_tokens=350, do_sample=True, temperature=0.7)
        )
        
        summary_text = self._extract_response_text(result).strip()
        
        # Log the response
        if connection_id:
            prompt_logger.log_summarization_prompt(
                connection_id=connection_id,
                conversation_history=conversation_history,
                response=summary_text
            )
        
        return summary_text

    def shutdown(self):
        """Shutdown the model manager and clean up resources."""
        self.logger.info("Shutting down thread executor...")
        self.executor.shutdown(wait=True)


# Global model manager instance
model_manager = ModelManager()

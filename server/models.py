"""
models.py
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


def sanitize_audio_placeholders(text: str) -> str:
    """Remove ALL audio placeholders from text"""
    if not text:
        return text
    
    import re
    # Remove all variations of audio placeholders
    text = re.sub(r'<\|audio\|?>', '', text)
    text = re.sub(r'<\|audio\|?', '', text)
    text = re.sub(r'<\|audio', '', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def ensure_one_audio_placeholder_last_user(turns: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    FIXED VERSION: Ensures exactly ONE <|audio|> placeholder in last USER turn.
    
    Critical fix: Must handle conversations with MULTIPLE turns correctly.
    Previous logic was breaking on longer conversations (5+ turns).
    """
    if not turns:
        return turns
    
    logger = logging.getLogger(__name__)
    
    # STEP 1: Clean ALL audio placeholders from ALL turns (critical!)
    for i, turn in enumerate(turns):
        if "content" in turn:
            original_content = turn["content"]
            turn["content"] = sanitize_audio_placeholders(turn["content"])
            
            if original_content != turn["content"]:
                logger.debug(f"Turn {i}: Removed audio placeholders")
    
    # STEP 2: Verify all placeholders are gone
    total_markers = sum(t.get("content", "").count("<|audio|>") for t in turns)
    if total_markers > 0:
        logger.error(f"CRITICAL: After sanitization, still found {total_markers} markers!")
        # This shouldn't happen, but if it does, log for debugging
        for i, turn in enumerate(turns):
            content = turn.get("content", "")
            if "<|audio|>" in content:
                logger.error(f"Turn {i} still has marker: {content[:100]}")
    
    # STEP 3: Find the LAST user turn (not first, not last assistant)
    last_user_idx = -1
    for i in range(len(turns) - 1, -1, -1):
        if turns[i].get("role") == "user":
            last_user_idx = i
            break
    
    logger.debug(f"Found last user turn at index: {last_user_idx}")
    
    # STEP 4: Handle case where no user turns exist
    if last_user_idx == -1:
        logger.warning("No user turns found - adding new user turn")
        turns.append({
            "role": "user",
            "content": "[Audio input] <|audio|>"
        })
        return turns
    
    # STEP 5: Add ONE placeholder to last user turn
    last_user_content = turns[last_user_idx].get("content", "").strip()
    if not last_user_content:
        last_user_content = "[Audio input]"
    
    # ADD placeholder to last user turn (NOT after, but AS PART OF the content)
    turns[last_user_idx]["content"] = last_user_content + " <|audio|>"
    
    logger.debug(f"Added placeholder to user turn {last_user_idx}: '{turns[last_user_idx]['content']}'")
    
    # STEP 6: Final verification - must have EXACTLY 1 marker
    total_markers = sum(t.get("content", "").count("<|audio|>") for t in turns)
    logger.info(f"Final marker count: {total_markers}")
    
    if total_markers != 1:
        logger.error(f"FATAL: Expected 1 marker, got {total_markers}!")
        logger.error("Conversation structure:")
        for i, turn in enumerate(turns):
            role = turn.get("role", "?")
            content = turn.get("content", "")[:60]
            markers = content.count("<|audio|>")
            logger.error(f"  Turn {i}: role={role}, markers={markers}, content='{content}...'")
        
        # EMERGENCY FALLBACK: Rebuild from scratch
        logger.warning("EMERGENCY FALLBACK: Rebuilding conversation structure")
        # Keep only system prompt and user content, rebuild
        new_turns = []
        for turn in turns:
            if turn.get("role") == "system":
                new_turns.append(turn)
        
        # Add user/assistant alternating without markers
        for turn in turns:
            if turn.get("role") in ["user", "assistant"]:
                new_turns.append(turn)
        
        # Add placeholder to last user
        for i in range(len(new_turns) - 1, -1, -1):
            if new_turns[i].get("role") == "user":
                content = new_turns[i].get("content", "").strip()
                new_turns[i]["content"] = content + " <|audio|>"
                break
        
        final_markers = sum(t.get("content", "").count("<|audio|>") for t in new_turns)
        logger.warning(f"After emergency rebuild: {final_markers} markers")
        
        if final_markers == 1:
            logger.warning("Emergency rebuild successful!")
            return new_turns
        else:
            logger.error("Emergency rebuild FAILED!")
            # Last resort: return error to caller
            return turns
    
    return turns


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
        """Process audio input using Ultravox pipeline with adaptive validation."""
        if not self.ultravox_pipeline:
            raise RuntimeError("Ultravox pipeline not loaded")
        
        try:
            import librosa
            import io
            from utils import is_valid_speech_audio
            from config import ULTRAVOX_MIN_ENERGY, ULTRAVOX_MIN_SPEECH_RATIO
            
            # Load audio
            audio_stream = io.BytesIO(audio_data)
            audio, sr = librosa.load(audio_stream, sr=16000)
            self.logger.info(f"Loaded audio: {len(audio)} samples at {sr}Hz")
            
            if len(audio) == 0:
                return "[AUDIO_EMPTY]"
            
            # TIER 2 VALIDATION
            is_valid_audio, stats = is_valid_speech_audio(
                audio_data,
                min_energy_threshold=ULTRAVOX_MIN_ENERGY,
                min_speech_ratio=ULTRAVOX_MIN_SPEECH_RATIO
            )
            
            if not is_valid_audio:
                reason = stats.get('reason', 'unknown')
                self.logger.warning(f"[ULTRAVOX] Audio rejected: {reason} (energy={stats.get('rms_energy', 0):.1f}, speech_ratio={stats.get('speech_ratio', 0):.3f})")
                
                # CRITICAL FIX: Clean up any existing audio placeholders from conversation context
                # This prevents accumulation of placeholders when audio is repeatedly rejected
                self._cleanup_audio_placeholders(conversation_turns)
                
                return "[AUDIO_REJECTED_BY_ULTRAVOX_VALIDATION]"
            
            # FIX: Reset conversation_turns to be a mutable copy
            conversation_turns = list(conversation_turns)  # Make copy
            
            # Check if conversation_turns already has proper structure (from get_conversation_context_for_ai)
            # If so, we need to add audio placeholder to the last user turn
            has_audio_placeholders = any("<|audio|>" in turn.get("content", "") for turn in conversation_turns)
            
            if not has_audio_placeholders:
                # This is processed context from get_conversation_context_for_ai, add a NEW user turn with placeholder
                # We should always add a new user turn for the current audio input, not modify existing turns
                conversation_turns.append({
                    "role": "user",
                    "content": "[Audio input] <|audio|>"
                })
                self.logger.info("Added new user turn with audio placeholder to processed context")
            else:
                # This is raw conversation data, use the existing function
                conversation_turns = ensure_one_audio_placeholder_last_user(conversation_turns)
            
            # Verify exactly 1 placeholder before sending to Ultravox
            total_audio_markers = sum(turn.get("content", "").count("<|audio|>") for turn in conversation_turns)
            
            if total_audio_markers != 1:
                self.logger.error(f"CRITICAL: Audio placeholder mismatch! Expected 1, found {total_audio_markers}")
                self.logger.error("Turn structure before sending:")
                for idx, turn in enumerate(conversation_turns):
                    role = turn.get("role", "?")
                    content = turn.get("content", "")
                    markers = content.count("<|audio|>")
                    self.logger.error(f"  Turn {idx}: role={role}, markers={markers}, content='{content[:80]}'")
                
                # CRITICAL FIX: Clean up placeholders and try to fix the issue
                self.logger.warning("Attempting to fix audio placeholder mismatch by cleaning up...")
                self._cleanup_audio_placeholders(conversation_turns)
                
                # Re-add exactly one placeholder by adding a new user turn
                conversation_turns.append({
                    "role": "user",
                    "content": "[Audio input] <|audio|>"
                })
                self.logger.info("Fixed: Added new user turn with single audio placeholder")
                
                # Verify fix
                total_audio_markers = sum(turn.get("content", "").count("<|audio|>") for turn in conversation_turns)
                if total_audio_markers != 1:
                    self.logger.error(f"Failed to fix audio placeholder issue. Still have {total_audio_markers} placeholders")
                    return "[PLACEHOLDER_VALIDATION_FAILED]"
                else:
                    self.logger.info("✅ Successfully fixed audio placeholder issue")
            
            # Ensure system prompt
            if not conversation_turns or conversation_turns[0].get("role") != "system":
                system_prompt = {
                    "role": "system",
                    "content": "You are Alexa, an HR recruiter from Novel Office calling Business Development Manager applicants. Speak naturally and professionally, as in a real phone call. Keep responses short, 1–2 sentences at a time. Do not use lists, bullets, emojis, stage directions, or overly formal prose; this is a live voice conversation. Always maintain your identity as Alexa and address the candidate appropriately."
                }
                conversation_turns = [system_prompt] + conversation_turns
            
            self.logger.info(f"[ULTRAVOX] Sending {len(conversation_turns)} turns to pipeline")
            for idx, turn in enumerate(conversation_turns):
                role = turn.get("role", "?")
                content = turn.get("content", "")
                markers = content.count("<|audio|>")
                self.logger.info(f"  Turn {idx}: role={role}, markers={markers}, content='{content[:70]}'")
            
            if connection_id:
                prompt_logger.log_ultravox_prompt(connection_id, audio_data, conversation_turns)
            
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
            
            self.logger.info("Ultravox inference completed successfully")
            response_text = self._extract_response_text(result)
            
            if connection_id:
                prompt_logger.log_ultravox_prompt(
                    connection_id, audio_data, conversation_turns, response_text
                )
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"Ultravox pipeline error: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return f"[ULTRAVOX_ERROR: {str(e)}]"
    
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
            
            # Load audio from bytes
            audio_stream = io.BytesIO(audio_data)
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

    def _cleanup_audio_placeholders(self, conversation_turns: List[Dict[str, str]]) -> None:
        """
        Clean up audio placeholders from conversation turns to prevent accumulation
        when audio is rejected by validation.
        
        Args:
            conversation_turns: List of conversation turns to clean up
        """
        if not conversation_turns:
            return
            
        self.logger.info("Cleaning up audio placeholders from conversation context")
        
        for turn in conversation_turns:
            content = turn.get("content", "")
            if "<|audio|>" in content:
                # Remove all audio placeholders from this turn
                cleaned_content = content.replace("<|audio|>", "").strip()
                
                # If content becomes empty after cleanup, set a default
                if not cleaned_content:
                    if turn.get("role") == "user":
                        cleaned_content = "[Audio input]"
                    else:
                        cleaned_content = "[Response]"
                
                turn["content"] = cleaned_content
                self.logger.info(f"Cleaned audio placeholder from {turn.get('role', 'unknown')} turn: '{content[:50]}...' -> '{cleaned_content[:50]}...'")
        
        # Verify cleanup
        total_placeholders = sum(turn.get("content", "").count("<|audio|>") for turn in conversation_turns)
        if total_placeholders == 0:
            self.logger.info("✅ Successfully cleaned all audio placeholders")
        else:
            self.logger.warning(f"⚠️ Still found {total_placeholders} audio placeholders after cleanup")

    def shutdown(self):
        """Shutdown the model manager and clean up resources."""
        self.logger.info("Shutting down thread executor...")
        self.executor.shutdown(wait=True)


# Global model manager instance
model_manager = ModelManager()

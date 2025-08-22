import os
import glob
from transformers import pipeline
from src.utils.logger import setup_logger
from config.settings import ULTRAVOX_SETTINGS, STORAGE_SETTINGS

class UltravoxSTT:
    def __init__(self):
        self.logger = setup_logger()
        # Use a more compatible model
        self.model_id = "openai/whisper-base"
        self.base_path = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/openai/whisper-base")
        self.language = ULTRAVOX_SETTINGS['language']
        
        # Check for model files in base path or subdirectories
        self.model_path = None
        if os.path.exists(self.base_path):
            # Look for config.json in base path or subdirectories
            config_files = glob.glob(os.path.join(self.base_path, "**/config.json"), recursive=True)
            if config_files:
                self.model_path = os.path.dirname(config_files[0])
                self.logger.info(f"Found model directory: {self.model_path}")
            else:
                self.logger.error(f"No config.json found in {self.base_path} or its subdirectories")
        
        # Try loading from local path
        if self.model_path:
            try:
                self.logger.info(f"Loading Ultravox model from {self.model_path}")
                self.stt = pipeline(
                    "automatic-speech-recognition",
                    model=self.model_path,
                    framework="pt",
                    device=-1  # CPU; use 0 for GPU if available
                )
                self.logger.info("Ultravox model loaded successfully from local path")
            except Exception as e:
                self.logger.error(f"Failed to load Ultravox model from local path: {e}")
                self.stt = None
        else:
            # Fallback to Hugging Face model ID
            try:
                self.logger.info(f"Attempting to load Ultravox model from Hugging Face: {self.model_id}")
                self.stt = pipeline(
                    "automatic-speech-recognition",
                    model=self.model_id,
                    framework="pt",
                    device=-1
                )
                self.logger.info("Ultravox model loaded successfully from Hugging Face")
            except Exception as e:
                self.logger.error(f"Failed to load from Hugging Face: {e}. Ensure HF_TOKEN is set for private repos.")
                self.stt = None

    def recognize(self, audio_path):
        """Recognize speech from audio file."""
        if self.stt is None:
            self.logger.error("Ultravox STT not initialized")
            return None

        try:
            if not os.path.exists(audio_path):
                self.logger.error(f"Audio file not found: {audio_path}")
                return None

            self.logger.info(f"Recognizing speech from {audio_path}")
            result = self.stt(audio_path, return_timestamps=False)
            text = result.get("text", "")
            self.logger.info(f"Recognized text: {text}")
            return text
        except Exception as e:
            self.logger.error(f"STT error: {e}")
            return None
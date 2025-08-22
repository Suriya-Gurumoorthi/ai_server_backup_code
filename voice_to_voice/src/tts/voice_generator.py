import os
from src.tts.piper_tts import PiperTTS
from src.core.audio_processor import AudioProcessor
from src.utils.logger import setup_logger
from config.settings import STORAGE_SETTINGS

class VoiceGenerator:
    def __init__(self):
        self.logger = setup_logger()
        self.tts = PiperTTS()
        self.audio_dir = STORAGE_SETTINGS['audio_dir']

    def generate_speech(self, text):
        """Generate speech from text."""
        import time
        import uuid
        # Use timestamp and UUID to ensure unique filenames
        timestamp = int(time.time() * 1000)
        unique_id = str(uuid.uuid4())[:8]
        output_path = f"output_{timestamp}_{unique_id}.wav"
        audio_path = self.tts.synthesize(text, output_path)
        # Don't play audio automatically - let the web interface handle it
        return audio_path
import os
from src.utils.logger import setup_logger
from src.core.audio_processor import AudioProcessor
from config.settings import STORAGE_SETTINGS, AUDIO_SETTINGS, CONVERSATION_SETTINGS

class VoiceSession:
    def __init__(self):
        self.logger = setup_logger()
        self.history = []
        self.audio_dir = STORAGE_SETTINGS['audio_dir']

    def get_audio_input(self, duration=AUDIO_SETTINGS['max_audio_length']):
        """Get audio input (from mic or file)."""
        processor = AudioProcessor()
        audio_path = processor.record_audio(duration)
        return audio_path

    def add_message(self, role, content):
        """Add a message to conversation history."""
        self.history.append({"role": role, "content": content})

    def get_history(self):
        """Return conversation history."""
        return self.history[-CONVERSATION_SETTINGS['context_window']:]  # Limit to context window
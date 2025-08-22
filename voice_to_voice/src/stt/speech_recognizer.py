from src.stt.ultravox_stt import UltravoxSTT
from src.utils.logger import setup_logger

class SpeechRecognizer:
    def __init__(self):
        self.logger = setup_logger()
        self.stt = UltravoxSTT()

    def recognize_speech(self, audio_path):
        """Recognize speech from audio file."""
        return self.stt.recognize(audio_path)
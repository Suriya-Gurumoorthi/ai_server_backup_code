import os
import wave
from src.utils.logger import setup_logger
from config.settings import PIPER_SETTINGS, STORAGE_SETTINGS

try:
    from piper import PiperVoice
except ImportError:
    PiperVoice = None

class PiperTTS:
    def __init__(self):
        self.logger = setup_logger()
        if PiperVoice is None:
            self.logger.warning("Piper TTS not available, using placeholder.")
            self.engine = None
        else:
            try:
                self.engine = PiperVoice.load(
                    model_path=PIPER_SETTINGS['model_path'],
                    config_path=PIPER_SETTINGS['config_path']
                )
                # Note: PiperVoice doesn't have set_quality method, using synthesis config instead
                self.logger.info("Piper TTS initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize Piper TTS: {e}")
                self.engine = None

    def synthesize(self, text, output_path):
        """Synthesize text to audio."""
        if self.engine is None:
            self.logger.error("Piper TTS not initialized, returning None.")
            return None
        try:
            output_path = os.path.join(STORAGE_SETTINGS['audio_dir'], output_path)
            with wave.open(output_path, "wb") as wav_file:
                self.engine.synthesize_wav(text, wav_file)
            self.logger.info(f"Synthesized audio saved to {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"TTS error: {e}")
            return None
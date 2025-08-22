import pytest
import os
from src.tts.piper_tts import PiperTTS
from config.settings import STORAGE_SETTINGS

def test_tts_synthesis():
    tts = PiperTTS()
    output_path = os.path.join(STORAGE_SETTINGS['audio_dir'], "test_output.wav")
    result = tts.synthesize("Hello, this is a test.", output_path)
    assert result is not None
    assert os.path.exists(result)
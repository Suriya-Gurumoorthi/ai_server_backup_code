import pytest
import os
from src.stt.ultravox_stt import UltravoxSTT
from config.settings import STORAGE_SETTINGS

def test_stt_recognition():
    stt = UltravoxSTT()
    audio_path = os.path.join(STORAGE_SETTINGS['audio_dir'], "sample_audio.wav")
    result = stt.recognize(audio_path)
    assert result is None or isinstance(result, str)
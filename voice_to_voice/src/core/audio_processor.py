import os
import sounddevice as sd
import scipy.io.wavfile as wavfile
from src.utils.audio_utils import save_audio
from src.utils.logger import setup_logger
from config.settings import AUDIO_SETTINGS, STORAGE_SETTINGS

class AudioProcessor:
    def __init__(self):
        self.logger = setup_logger()
        self.sample_rate = AUDIO_SETTINGS['sample_rate']
        self.channels = AUDIO_SETTINGS['channels']
        self.audio_dir = STORAGE_SETTINGS['audio_dir']

    def record_audio(self, duration=AUDIO_SETTINGS['max_audio_length']):
        """Record audio from microphone."""
        try:
            # List available audio devices
            devices = sd.query_devices()
            self.logger.info(f"Available audio devices: {len(devices)}")
            
            # Find default input device
            default_input = sd.query_devices(kind='input')
            self.logger.info(f"Default input device: {default_input}")
            
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=AUDIO_SETTINGS['format'],
                device=default_input['index'] if default_input else None
            )
            sd.wait()  # Wait until recording is finished
            output_path = os.path.join(self.audio_dir, f"input_{id(self)}.wav")
            save_audio(recording, output_path, self.sample_rate)
            self.logger.info(f"Recorded audio to {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Error recording audio: {e}")
            return None

    def play_audio(self, audio_path):
        """Play an audio file."""
        try:
            # Don't play audio automatically - let the web interface handle it
            self.logger.info(f"Audio ready for playback: {audio_path}")
            return audio_path
        except Exception as e:
            self.logger.error(f"Error preparing audio: {e}")
            raise
#!/usr/bin/env python3
"""
Standalone TTS script using Piper TTS directly.
This uses the same Piper TTS system that's already working in your server.
"""

import sys
import os
import asyncio
import logging

# Add the server directory to the path
sys.path.append('/home/novel/server')

# Change to server directory
os.chdir('/home/novel/server')

try:
    from piper import PiperVoice
    import torch
    import torchaudio as ta
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Please install: pip install piper-tts torch torchaudio")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_tts_audio(text: str, output_file: str = "output.wav"):
    """Generate TTS audio from text using Piper TTS."""
    try:
        # Check if Piper model files exist
        model_path = "en_US-lessac-medium.onnx"
        config_path = "en_US-lessac-medium.json"
        
        if not os.path.exists(model_path):
            print(f"❌ Piper model file not found: {model_path}")
            print("Please download the Piper model files to the current directory")
            return False
            
        if not os.path.exists(config_path):
            print(f"❌ Piper config file not found: {config_path}")
            print("Please download the Piper model files to the current directory")
            return False
        
        print(f"🎤 Loading Piper TTS model...")
        
        # Load Piper voice
        voice = PiperVoice.load(model_path, config_path)
        
        print(f"🎤 Generating TTS for: '{text}'")
        
        # Generate audio
        audio = voice.synthesize(text)
        
        # Save audio to file
        ta.save(output_file, audio, voice.sample_rate)
        
        # Check if file was created
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"✅ TTS audio saved to: {output_file}")
            print(f"📊 Audio size: {file_size} bytes")
            print(f"🎵 Sample rate: {voice.sample_rate} Hz")
            return True
        else:
            print("❌ Failed to create audio file")
            return False
            
    except Exception as e:
        print(f"❌ Error generating TTS: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to test TTS generation."""
    # Test text (same as your original chatterbox example)
    text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
    
    print("🎤 Starting TTS generation with Piper TTS...")
    
    # Generate first audio file
    success1 = generate_tts_audio(text, "test-1.wav")
    
    if success1:
        # Generate second audio file (like your original script)
        success2 = generate_tts_audio(text, "test-2.wav")
        
        if success2:
            print("🎉 Both TTS files generated successfully!")
            print("📁 Files created:")
            print("   - test-1.wav")
            print("   - test-2.wav")
        else:
            print("❌ Failed to generate second audio file")
    else:
        print("❌ Failed to generate first audio file")

if __name__ == "__main__":
    main()




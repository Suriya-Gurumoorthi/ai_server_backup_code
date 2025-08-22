#!/usr/bin/env python3
"""
Quick script to process different audio files without reloading the Ultravox model.
Just change the audio_path variable and run this script.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.processors.audio_processor import process_audio_file
from src.models.ultravox_model import is_model_loaded

# ===========================================
# CHANGE THIS PATH TO YOUR AUDIO FILE
# ===========================================
audio_path = "Audios/ATSID00897933_introduction.wav"
# ===========================================

def main():
    print("="*60)
    print("ULTRAVOX AUDIO EVALUATION")
    print("="*60)
    print(f"Processing: {audio_path}")
    
    # Check model status
    if is_model_loaded():
        print("✅ Model already loaded - will reuse from VRAM")
    else:
        print("🔄 Model not loaded - will load into VRAM")
    
    print("="*60)
    
    # Process the audio file
    result = process_audio_file(audio_path)
    
    if result:
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(result)
        print("="*60)
    else:
        print("❌ Failed to process audio file")

if __name__ == "__main__":
    main() 
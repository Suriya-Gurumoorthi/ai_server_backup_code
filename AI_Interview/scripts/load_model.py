#!/usr/bin/env python3
"""
Script to pre-load the Ultravox model into VRAM.
Run this once to load the model, then use quick_process.py for processing audio files.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.ultravox_model import get_pipe

def main():
    print("="*60)
    print("LOADING ULTRAVOX MODEL INTO VRAM")
    print("="*60)
    
    # Load the model
    pipe = get_pipe()
    
    print("="*60)
    print("✅ MODEL SUCCESSFULLY LOADED INTO VRAM")
    print("✅ You can now run quick_process.py without model reloading")
    print("="*60)

if __name__ == "__main__":
    main() 
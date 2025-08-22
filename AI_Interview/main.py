#!/usr/bin/env python3
"""
Main application for AI Interview Evaluation System.
Provides a clean interface to evaluate interview audio files.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.processors.audio_processor import process_audio_file
from src.models.ultravox_model import is_model_loaded
from configs.config import PATHS, AUDIO_CONFIG

def validate_audio_file(audio_path):
    """Validate if the audio file exists and is supported"""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    file_ext = Path(audio_path).suffix.lower()
    if file_ext not in AUDIO_CONFIG["supported_formats"]:
        raise ValueError(f"Unsupported audio format: {file_ext}. Supported formats: {AUDIO_CONFIG['supported_formats']}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="AI Interview Evaluation System")
    parser.add_argument("audio_file", help="Path to the audio file to evaluate")
    parser.add_argument("--output", "-o", help="Output file path for results (optional)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    try:
        # Validate audio file
        validate_audio_file(args.audio_file)
        
        print("="*70)
        print("🎤 AI INTERVIEW EVALUATION SYSTEM")
        print("="*70)
        print(f"📁 Audio File: {args.audio_file}")
        
        # Check model status
        if is_model_loaded():
            print("✅ Model already loaded - will reuse from VRAM")
        else:
            print("🔄 Model not loaded - will load into VRAM")
        
        print("="*70)
        
        # Process the audio file
        result = process_audio_file(args.audio_file)
        
        if result:
            print("\n" + "="*70)
            print("📊 EVALUATION RESULTS")
            print("="*70)
            print(result)
            print("="*70)
            
            # Save results if output file specified
            if args.output:
                os.makedirs(os.path.dirname(args.output), exist_ok=True)
                with open(args.output, 'w') as f:
                    f.write(str(result))
                print(f"💾 Results saved to: {args.output}")
        else:
            print("❌ Failed to process audio file")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
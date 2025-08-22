#!/usr/bin/env python3
"""
Model Storage Information - Shows where different models are stored and their sizes
"""

import os
import subprocess
from pathlib import Path
import humanize

def get_directory_size(directory):
    """Get the size of a directory in bytes"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception as e:
        print(f"Error calculating size for {directory}: {e}")
    return total_size

def check_ollama_models():
    """Check Ollama models location and size"""
    print("=== Ollama Models (Text-based LLMs) ===")
    
    ollama_path = os.path.expanduser("~/.ollama/models")
    if os.path.exists(ollama_path):
        print(f"📍 Location: {ollama_path}")
        
        # Get total size
        total_size = get_directory_size(ollama_path)
        print(f"📊 Total size: {humanize.naturalsize(total_size)}")
        
        # List individual models
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if result.returncode == 0:
                print("\n📋 Installed models:")
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 2:
                            model_name = parts[0]
                            size_str = parts[1] if len(parts) > 1 else "Unknown"
                            print(f"   - {model_name}: {size_str}")
        except Exception as e:
            print(f"   Error listing models: {e}")
    else:
        print("❌ Ollama models directory not found")
    
    print()

def check_speechbrain_models():
    """Check SpeechBrain models location and size"""
    print("=== SpeechBrain Models (Audio/Speech) ===")
    
    # Check current directory for pretrained_models
    current_dir = os.getcwd()
    pretrained_path = os.path.join(current_dir, "pretrained_models")
    
    print(f"📍 Location: {pretrained_path}")
    
    if os.path.exists(pretrained_path):
        total_size = get_directory_size(pretrained_path)
        print(f"📊 Total size: {humanize.naturalsize(total_size)}")
        
        # List individual models
        print("\n📋 Installed models:")
        for item in os.listdir(pretrained_path):
            item_path = os.path.join(pretrained_path, item)
            if os.path.isdir(item_path):
                size = get_directory_size(item_path)
                print(f"   - {item}: {humanize.naturalsize(size)}")
    else:
        print("📋 No SpeechBrain models installed yet")
        print("   (Will be created when you run the emotion recognition setup)")
    
    print()

def show_model_comparison():
    """Show comparison between different model types"""
    print("=== Model Type Comparison ===")
    print()
    print("🔤 Text-based LLMs (Ollama):")
    print("   📍 Storage: ~/.ollama/models/")
    print("   📊 Size: 3GB - 40GB+ per model")
    print("   🎯 Use: Text generation, conversations, coding")
    print("   📦 Examples: llama3.2:3b, llama3.2:8b, codellama:7b")
    print()
    print("🎵 Audio/Speech Models (SpeechBrain):")
    print("   📍 Storage: ./pretrained_models/ (current directory)")
    print("   📊 Size: 100MB - 1GB per model")
    print("   🎯 Use: Speech recognition, emotion detection, audio processing")
    print("   📦 Examples: emotion-recognition-wav2vec2-IEMOCAP")
    print()

def estimate_download_size():
    """Estimate download size for SpeechBrain emotion model"""
    print("=== Download Size Estimation ===")
    print()
    print("🎭 speechbrain/emotion-recognition-wav2vec2-IEMOCAP:")
    print("   📊 Estimated size: ~500MB - 1GB")
    print("   📁 Components:")
    print("      - Model weights: ~300-500MB")
    print("      - Configuration files: ~1-5MB")
    print("      - Vocabulary files: ~1-10MB")
    print("      - Preprocessing pipeline: ~10-50MB")
    print()
    print("💾 Storage location: ./pretrained_models/emotion-recognition-wav2vec2-IEMOCAP/")
    print("⏱️  Download time: 2-10 minutes (depending on internet speed)")
    print()

def main():
    print("🤖 Model Storage Information\n")
    
    check_ollama_models()
    check_speechbrain_models()
    show_model_comparison()
    estimate_download_size()
    
    print("=== Next Steps ===")
    print("1. To download SpeechBrain emotion model:")
    print("   python speechbrain_emotion_setup.py")
    print()
    print("2. To check Ollama models:")
    print("   ollama list")
    print()
    print("3. To see storage usage:")
    print("   du -sh ~/.ollama/models/")
    print("   du -sh ./pretrained_models/")

if __name__ == "__main__":
    main() 
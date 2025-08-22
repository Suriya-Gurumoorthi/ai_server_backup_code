#!/usr/bin/env python3
"""
Simple Model Storage Information - Shows where different models are stored
"""

import os
import subprocess

def format_size(size_bytes):
    """Convert bytes to human readable format"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"

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

def main():
    print("🤖 Model Storage Information\n")
    
    # Check Ollama models
    print("=== Ollama Models (Text-based LLMs) ===")
    ollama_path = os.path.expanduser("~/.ollama/models")
    if os.path.exists(ollama_path):
        print(f"📍 Location: {ollama_path}")
        total_size = get_directory_size(ollama_path)
        print(f"📊 Total size: {format_size(total_size)}")
        
        # List models
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if result.returncode == 0:
                print("\n📋 Installed models:")
                lines = result.stdout.strip().split('\n')[1:]
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
    
    # Check SpeechBrain models
    print("=== SpeechBrain Models (Audio/Speech) ===")
    current_dir = os.getcwd()
    pretrained_path = os.path.join(current_dir, "pretrained_models")
    print(f"📍 Location: {pretrained_path}")
    
    if os.path.exists(pretrained_path):
        total_size = get_directory_size(pretrained_path)
        print(f"📊 Total size: {format_size(total_size)}")
        print("\n📋 Installed models:")
        for item in os.listdir(pretrained_path):
            item_path = os.path.join(pretrained_path, item)
            if os.path.isdir(item_path):
                size = get_directory_size(item_path)
                print(f"   - {item}: {format_size(size)}")
    else:
        print("📋 No SpeechBrain models installed yet")
        print("   (Will be created when you run the emotion recognition setup)")
    
    print()
    
    # Model comparison
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
    
    # Download estimation
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
    
    print("=== Next Steps ===")
    print("1. To download SpeechBrain emotion model:")
    print("   python speechbrain_emotion_setup.py")
    print()
    print("2. To check Ollama models:")
    print("   ollama list")

if __name__ == "__main__":
    main() 
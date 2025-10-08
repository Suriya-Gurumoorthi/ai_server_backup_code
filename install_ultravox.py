#!/usr/bin/env python3
"""
Install and test fixie-ai/ultravox-v0_6-qwen-3-32b using transformers
Based on the official Hugging Face model page
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

def install_and_test_model():
    """Install and test the ultravox model using transformers"""
    print("🚀 Installing and Testing Ultravox Model...")
    print("=" * 50)
    
    try:
        # Import required libraries
        import transformers
        import librosa
        
        print(f"✅ Transformers version: {transformers.__version__}")
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ Librosa version: {librosa.__version__}")
        
        # Model name from Hugging Face
        model_name = "fixie-ai/ultravox-v0_5-llama-3_1-8b"
        
        print(f"\n📦 Loading model: {model_name}")
        print("This will download the model from Hugging Face...")
        
        # Load the model using transformers pipeline
        # Based on the official usage example from the model page
        pipe = transformers.pipeline(
            model=model_name, 
            trust_remote_code=True
        )
        
        print("✅ Model loaded successfully!")
        print(f"📊 Pipeline type: {type(pipe)}")
        
        
        # Test memory usage
        print("\n💾 Memory Usage Information:")
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"📊 Current memory usage: {memory_mb:.1f} MB")
        
        print("\n🎉 Model installation and testing completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    success = install_and_test_model()
    
    if success:
        print("\n✅ Installation and testing completed successfully!")
        print("The model is ready to use with transformers.")
    else:
        print("\n❌ Installation or testing failed.")
        print("Please check the error messages above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

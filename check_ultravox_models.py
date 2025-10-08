#!/usr/bin/env python3
"""
Check available ultravox models and their sizes
"""

import requests
import json

def check_ultravox_models():
    """Check available ultravox models"""
    print("🔍 Checking Available Ultravox Models...")
    print("=" * 50)
    
    # Known ultravox models
    models = [
        "fixie-ai/ultravox-v0_6-qwen-3-8b",
        "fixie-ai/ultravox-v0_6-qwen-3-14b", 
        "fixie-ai/ultravox-v0_6-qwen-3-32b",
        "fixie-ai/ultravox-v0_6-llama-3_1-8b",
        "fixie-ai/ultravox-v0_6-gemma-3-27b"
    ]
    
    for model in models:
        try:
            url = f"https://huggingface.co/api/models/{model}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                size_gb = data.get('safetensors', {}).get('total', 0) / (1024**3)
                print(f"✅ {model}")
                print(f"   Size: {size_gb:.1f} GB")
                print(f"   Downloads: {data.get('downloads', 'N/A')}")
                print()
            else:
                print(f"❌ {model} - Not found")
        except Exception as e:
            print(f"❌ Error checking {model}: {e}")

if __name__ == "__main__":
    check_ultravox_models()

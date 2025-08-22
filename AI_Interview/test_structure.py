#!/usr/bin/env python3
"""
Test script to verify the modular structure is working correctly.
This script tests imports and basic functionality without loading the model.
"""

import sys
import os

def test_imports():
    """Test that all modules can be imported correctly"""
    print("🧪 Testing imports...")
    
    try:
        # Test src imports
        from src.models.ultravox_model import get_pipe, is_model_loaded, model_instance
        print("✅ src.models.ultravox_model imported successfully")
        
        from src.processors.audio_processor import process_audio_file, create_evaluation_prompt
        print("✅ src.processors.audio_processor imported successfully")
        
        from configs.config import MODEL_CONFIG, AUDIO_CONFIG, PATHS
        print("✅ configs.config imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_model_status():
    """Test model loading status function"""
    print("\n🧪 Testing model status...")
    
    try:
        from src.models.ultravox_model import is_model_loaded
        
        # Test initial status (should be False)
        status = is_model_loaded()
        print(f"✅ Model loaded status: {status} (expected: False)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model status: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print("\n🧪 Testing configuration...")
    
    try:
        from configs.config import MODEL_CONFIG, AUDIO_CONFIG, PATHS
        
        print(f"✅ Model ID: {MODEL_CONFIG['model_id']}")
        print(f"✅ Supported audio formats: {AUDIO_CONFIG['supported_formats']}")
        print(f"✅ Audio directory: {PATHS['audio_dir']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing configuration: {e}")
        return False

def main():
    """Main test function"""
    print("="*60)
    print("🏗️ TESTING MODULAR STRUCTURE")
    print("="*60)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test model status
    model_ok = test_model_status()
    
    # Test configuration
    config_ok = test_config()
    
    print("\n" + "="*60)
    print("📊 TEST RESULTS")
    print("="*60)
    print(f"Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Model Status: {'✅ PASS' if model_ok else '❌ FAIL'}")
    print(f"Configuration: {'✅ PASS' if config_ok else '❌ FAIL'}")
    
    if all([imports_ok, model_ok, config_ok]):
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Modular structure is working correctly")
        print("\n📝 Next steps:")
        print("1. Activate your virtual environment: source venv/bin/activate")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Test model loading: python scripts/load_model.py")
        print("4. Process audio: python main.py Audios/your_audio.wav")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
    
    print("="*60)

if __name__ == "__main__":
    main() 
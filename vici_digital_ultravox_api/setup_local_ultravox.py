#!/usr/bin/env python3
"""
Setup script for local Ultravox model integration
This script helps configure and test the local Ultravox model
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_system_requirements():
    """Check if system meets requirements for local Ultravox"""
    logger.info("Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"   CUDA version: {torch.version.cuda}")
    else:
        logger.warning("⚠️  CUDA not available, will use CPU (slower)")
    
    # Check available memory
    if cuda_available:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"   GPU memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 8:
            logger.warning("⚠️  GPU memory less than 8GB, may cause issues with large models")
    
    return True

def check_ultravox_installation():
    """Check if Ultravox is properly installed"""
    logger.info("Checking Ultravox installation...")
    
    try:
        import ultravox
        logger.info("✅ Ultravox package found")
        
        # Try to get version
        try:
            version = ultravox.__version__
            logger.info(f"   Version: {version}")
        except:
            logger.info("   Version: Unknown")
        
        return True
        
    except ImportError:
        logger.error("❌ Ultravox package not found")
        logger.info("   Install with: pip install ultravox")
        return False

def check_model_path(model_path):
    """Check if Ultravox model is available at the specified path"""
    logger.info(f"Checking Ultravox model at: {model_path}")
    
    # Check if it's a local path
    if os.path.exists(model_path):
        logger.info("✅ Local model path exists")
        return True
    
    # Check if it's a Hugging Face model
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, local_files_only=False)
        logger.info("✅ Hugging Face model found")
        logger.info(f"   Model type: {config.model_type}")
        return True
    except Exception as e:
        logger.error(f"❌ Model not found: {e}")
        return False

def download_model_if_needed(model_path):
    """Download the Ultravox model if not already present"""
    logger.info(f"Checking if model needs to be downloaded: {model_path}")
    
    # If it's a local path and exists, no need to download
    if os.path.exists(model_path):
        logger.info("✅ Model already exists locally")
        return True
    
    # If it's a Hugging Face model, try to download
    try:
        logger.info("Downloading model from Hugging Face...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("✅ Tokenizer downloaded")
        
        # Download model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        logger.info("✅ Model downloaded")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download model: {e}")
        return False

def test_model_loading(model_path):
    """Test loading the Ultravox model"""
    logger.info("Testing model loading...")
    
    try:
        from ultravox import Ultravox
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model on device: {device}")
        
        model = Ultravox.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        logger.info("✅ Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

def create_environment_file():
    """Create a .env file with recommended settings"""
    logger.info("Creating environment configuration...")
    
    env_content = f"""# Ultravox Bridge Configuration
BRIDGE_SECRET_KEY=your-super-secret-key-change-this
ALLOWED_IPS=192.168.1.100,192.168.1.101
ENABLE_AUTH=true

# Ultravox Model Configuration
ULTRAVOX_MODEL_PATH=fixie-ai/ultravox
ULTRAVOX_DEVICE={"cuda" if torch.cuda.is_available() else "cpu"}

# Audio Processing
VICIDIAL_SAMPLE_RATE=8000
ULTRAVOX_SAMPLE_RATE=48000
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    logger.info("✅ Created .env file with default configuration")
    logger.info("   Edit .env file to customize settings")

def main():
    """Main setup function"""
    logger.info("Ultravox Local Setup Script")
    logger.info("=" * 40)
    
    # Get model path from user or use default
    model_path = input("Enter Ultravox model path (default: fixie-ai/ultravox): ").strip()
    if not model_path:
        model_path = "fixie-ai/ultravox"
    
    # Run checks
    checks_passed = 0
    total_checks = 5
    
    # Check 1: System requirements
    if check_system_requirements():
        checks_passed += 1
    
    # Check 2: Ultravox installation
    if check_ultravox_installation():
        checks_passed += 1
    
    # Check 3: Model availability
    if check_model_path(model_path):
        checks_passed += 1
    
    # Check 4: Download model if needed
    if download_model_if_needed(model_path):
        checks_passed += 1
    
    # Check 5: Test model loading
    if test_model_loading(model_path):
        checks_passed += 1
    
    # Create environment file
    create_environment_file()
    
    # Summary
    logger.info("\n" + "=" * 40)
    logger.info("SETUP SUMMARY")
    logger.info("=" * 40)
    logger.info(f"Checks passed: {checks_passed}/{total_checks}")
    
    if checks_passed == total_checks:
        logger.info("🎉 Setup completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Edit .env file with your settings")
        logger.info("2. Start the bridge: python openai_local.py")
        logger.info("3. Test with: python test_remote_setup.py <bridge_ip>")
    else:
        logger.warning("⚠️  Some checks failed. Please resolve issues before proceeding.")
        logger.info("\nCommon solutions:")
        logger.info("- Install missing packages: pip install -r requirements_local.txt")
        logger.info("- Check internet connection for model download")
        logger.info("- Ensure sufficient disk space for model")
        logger.info("- Verify CUDA installation if using GPU")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)





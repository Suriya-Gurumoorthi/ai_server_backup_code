#!/bin/bash

echo "=== SpeechBrain Emotion Recognition Model Installation ==="
echo "This script will install SpeechBrain and download the emotion recognition model."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python3 first."
    exit 1
fi

echo "✅ Python3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3 first."
    exit 1
fi

echo "✅ pip3 found: $(pip3 --version)"

# Install required packages
echo
echo "Installing required packages..."

# Install PyTorch first (CPU version for compatibility)
echo "Installing PyTorch..."
pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install SpeechBrain
echo "Installing SpeechBrain..."
pip3 install speechbrain

# Install other dependencies
echo "Installing other dependencies..."
pip3 install requests numpy

echo
echo "✅ All packages installed successfully!"

# Test the installation
echo
echo "Testing installation..."
python3 -c "
import torch
import torchaudio
import speechbrain
print('✅ PyTorch version:', torch.__version__)
print('✅ TorchAudio version:', torchaudio.__version__)
print('✅ SpeechBrain version:', speechbrain.__version__)
print('✅ Installation successful!')
"

if [ $? -eq 0 ]; then
    echo
    echo "🎉 Installation completed successfully!"
    echo
    echo "Next steps:"
    echo "1. Run the setup script: python3 speechbrain_emotion_setup.py"
    echo "2. Or use the model directly in your code"
    echo
    echo "Example usage:"
    echo "python3 speechbrain_emotion_setup.py"
else
    echo
    echo "❌ Installation test failed. Please check the error messages above."
    exit 1
fi 
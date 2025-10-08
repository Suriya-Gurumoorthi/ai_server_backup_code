#!/bin/bash

echo "🎤 Installing Voice Cloning Dependencies"
echo "========================================"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Not in a virtual environment. Please activate ultravox_env first:"
    echo "   source ultravox_env/bin/activate"
    exit 1
fi

echo "✅ Virtual environment detected: $VIRTUAL_ENV"

# Install basic dependencies
echo "📦 Installing basic dependencies..."
pip install -r requirements_voice_clone.txt

# Install Coqui TTS for voice cloning
echo "🎭 Installing Coqui TTS for voice cloning..."
pip install TTS

# Check for ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  ffmpeg not found. Installing..."
    sudo apt update
    sudo apt install -y ffmpeg
else
    echo "✅ ffmpeg already installed"
fi

# Test installations
echo "🧪 Testing installations..."

# Test Piper TTS
python3 -c "from piper import PiperVoice; print('✅ Piper TTS installed successfully')" 2>/dev/null || echo "❌ Piper TTS installation failed"

# Test Coqui TTS
python3 -c "import TTS; print('✅ Coqui TTS installed successfully')" 2>/dev/null || echo "❌ Coqui TTS installation failed"

# Test librosa
python3 -c "import librosa; print('✅ Librosa installed successfully')" 2>/dev/null || echo "❌ Librosa installation failed"

echo ""
echo "🎉 Installation Complete!"
echo ""
echo "Usage Examples:"
echo "==============="
echo ""
echo "1. Basic audio analysis with TTS response:"
echo "   python3 voice_clone_tts_response.py input.wav"
echo ""
echo "2. Audio analysis with voice cloning:"
echo "   python3 voice_clone_tts_response.py input.wav reference_voice.wav my_voice"
echo ""
echo "3. Test with existing audio files:"
echo "   python3 voice_clone_tts_response.py voice_to_voice/custom_test.wav"
echo ""
echo "Output files will be saved in: voice_clone_output/session_YYYYMMDD_HHMMSS/"


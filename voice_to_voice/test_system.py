#!/usr/bin/env python3
"""
Test script for the Voice-to-Voice AI System
"""

import requests
import json
import time
import os
from pathlib import Path

def test_api_status():
    """Test the API status endpoint."""
    print("Testing API status...")
    try:
        response = requests.get("http://localhost:5000/api/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Status: {data}")
            return True
        else:
            print(f"❌ API Status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API Status error: {e}")
        return False

def test_web_interface():
    """Test the web interface."""
    print("Testing web interface...")
    try:
        response = requests.get("http://localhost:5000/")
        if response.status_code == 200:
            print("✅ Web interface accessible")
            return True
        else:
            print(f"❌ Web interface failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Web interface error: {e}")
        return False

def test_tts_functionality():
    """Test TTS functionality."""
    print("Testing TTS functionality...")
    try:
        from src.tts.voice_generator import VoiceGenerator
        tts = VoiceGenerator()
        test_text = "Hello, this is a test of the text-to-speech system."
        
        # Test TTS without audio playback
        audio_path = tts.tts.synthesize(test_text, f"test_output_{id(tts)}.wav")
        if audio_path and os.path.exists(audio_path):
            print(f"✅ TTS working: {audio_path}")
            return True
        else:
            print("❌ TTS failed: No audio file generated")
            return False
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return False

def test_stt_functionality():
    """Test STT functionality."""
    print("Testing STT functionality...")
    try:
        from src.stt.speech_recognizer import SpeechRecognizer
        stt = SpeechRecognizer()
        
        # Test with existing audio file if available
        test_audio = "test.wav"
        if os.path.exists(test_audio):
            text = stt.recognize_speech(test_audio)
            if text:
                print(f"✅ STT working: '{text}'")
                return True
            else:
                print("❌ STT failed: No text recognized")
                return False
        else:
            print("⚠️  No test audio file found, skipping STT test")
            return True
    except Exception as e:
        print(f"❌ STT error: {e}")
        return False

def test_conversation_flow():
    """Test conversation flow."""
    print("Testing conversation flow...")
    try:
        from src.conversation.conversation_flow import ConversationFlow
        conv = ConversationFlow()
        test_input = "Hello, how are you?"
        response = conv.generate_response(test_input, [])
        if response:
            print(f"✅ Conversation flow working: '{response}'")
            return True
        else:
            print("❌ Conversation flow failed: No response generated")
            return False
    except Exception as e:
        print(f"❌ Conversation flow error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Voice-to-Voice AI System Test Suite")
    print("=" * 50)
    
    tests = [
        test_api_status,
        test_web_interface,
        test_tts_functionality,
        test_stt_functionality,
        test_conversation_flow
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs for details.")
    
    return passed == total

if __name__ == "__main__":
    main() 
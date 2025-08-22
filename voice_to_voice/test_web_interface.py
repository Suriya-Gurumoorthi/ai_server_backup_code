#!/usr/bin/env python3
"""
Test script for the web interface functionality
"""

import requests
import json
import time

def test_web_interface():
    """Test the web interface endpoints."""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Web Interface")
    print("=" * 40)
    
    # Test 1: Check if server is running
    print("1. Testing server status...")
    try:
        response = requests.get(f"{base_url}/api/status")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Server running: {data}")
        else:
            print(f"   ❌ Server error: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Server connection failed: {e}")
        return False
    
    # Test 2: Check web interface
    print("2. Testing web interface...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("   ✅ Web interface accessible")
            if "Voice-to-Voice AI System" in response.text:
                print("   ✅ Correct HTML content")
            else:
                print("   ⚠️  HTML content may be incomplete")
        else:
            print(f"   ❌ Web interface error: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Web interface failed: {e}")
        return False
    
    # Test 3: Check static files
    print("3. Testing static files...")
    static_files = [
        "/static/css/style.css",
        "/static/js/voice_interface.js",
        "/static/js/websocket_client.js"
    ]
    
    for file_path in static_files:
        try:
            response = requests.get(f"{base_url}{file_path}")
            if response.status_code == 200:
                print(f"   ✅ {file_path} accessible")
            else:
                print(f"   ❌ {file_path} not found: {response.status_code}")
        except Exception as e:
            print(f"   ❌ {file_path} error: {e}")
    
    print("\n" + "=" * 40)
    print("🎉 Web interface tests completed!")
    print("\n📋 Next Steps:")
    print("1. Open your browser and go to: http://127.0.0.1:5000")
    print("2. Select an AI role from the sidebar")
    print("3. Click 'Start Conversation'")
    print("4. Allow microphone access when prompted")
    print("5. Start speaking to test the voice interface")
    
    return True

if __name__ == "__main__":
    test_web_interface() 
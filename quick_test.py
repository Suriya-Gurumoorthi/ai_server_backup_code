#!/usr/bin/env python3
"""
Quick test script for Ultravox WebSocket server
Tests basic functionality to identify issues quickly
"""

import asyncio
import websockets
import struct
import numpy as np
import time

async def test_basic_connection():
    """Test basic connection and text input"""
    print("🔗 Testing basic connection...")
    
    try:
        async with websockets.connect("ws://localhost:8765") as websocket:
            print("✅ Connected successfully!")
            
            # Test text input
            print("💬 Testing text input...")
            await websocket.send("Hello, this is a test")
            
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10)
                print(f"✅ Text response: {response}")
                return True
            except asyncio.TimeoutError:
                print("⏰ Text response timeout")
                return False
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

async def test_audio_processing():
    """Test audio processing with simple audio"""
    print("\n🎵 Testing audio processing...")
    
    try:
        async with websockets.connect("ws://localhost:8765") as websocket:
            print("✅ Connected for audio test!")
            
            # Generate simple test audio (0.5 seconds, 440Hz)
            sample_rate = 16000
            duration = 0.5
            frequency = 440
            
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_signal = np.sin(2 * np.pi * frequency * t)
            audio_int16 = (audio_signal * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            # Create WAV header
            wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
                b'RIFF',
                36 + len(audio_bytes),
                b'WAVE',
                b'fmt ',
                16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
                b'data',
                len(audio_bytes)
            )
            
            wav_data = wav_header + audio_bytes
            
            print(f"📤 Sending {len(wav_data)} bytes of audio...")
            start_time = time.time()
            await websocket.send(wav_data)
            
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=15)
                processing_time = time.time() - start_time
                print(f"✅ Audio response: {response}")
                print(f"⏱️  Processing time: {processing_time:.2f}s")
                return True
            except asyncio.TimeoutError:
                print("⏰ Audio response timeout")
                return False
                
    except Exception as e:
        print(f"❌ Audio test failed: {e}")
        return False

async def test_error_handling():
    """Test error handling with invalid data"""
    print("\n🚨 Testing error handling...")
    
    try:
        async with websockets.connect("ws://localhost:8765") as websocket:
            print("✅ Connected for error test!")
            
            # Send invalid data
            invalid_data = b"this is not audio data"
            print("📤 Sending invalid data...")
            await websocket.send(invalid_data)
            
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10)
                print(f"✅ Error response: {response}")
                return True
            except asyncio.TimeoutError:
                print("⏰ Error response timeout")
                return False
                
    except Exception as e:
        print(f"❌ Error test failed: {e}")
        return False

async def main():
    """Run quick tests"""
    print("🚀 Quick Ultravox Server Test")
    print("=" * 40)
    
    # Test 1: Basic connection and text
    text_success = await test_basic_connection()
    
    # Test 2: Audio processing
    audio_success = await test_audio_processing()
    
    # Test 3: Error handling
    error_success = await test_error_handling()
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 QUICK TEST RESULTS:")
    print(f"Text Processing: {'✅ PASS' if text_success else '❌ FAIL'}")
    print(f"Audio Processing: {'✅ PASS' if audio_success else '❌ FAIL'}")
    print(f"Error Handling: {'✅ PASS' if error_success else '❌ FAIL'}")
    
    total_success = sum([text_success, audio_success, error_success])
    print(f"\nOverall: {total_success}/3 tests passed")
    
    if total_success == 3:
        print("🎉 All tests passed! Server is working well.")
    elif total_success >= 2:
        print("⚠️  Most tests passed, minor issues detected.")
    else:
        print("❌ Multiple issues detected, server needs attention.")

if __name__ == "__main__":
    asyncio.run(main())


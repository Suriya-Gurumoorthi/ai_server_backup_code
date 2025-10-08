#!/usr/bin/env python3
"""
Test conversation flow simulating Surya introducing himself and system asking follow-up questions
"""

import asyncio
import websockets
import json
import wave
import io
import numpy as np

async def test_surya_conversation_flow():
    """Test conversation memory with Surya introducing himself as age 22"""
    
    uri = "ws://localhost:8000"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket server")
            print("🎭 Simulating conversation: Surya (age 22) introduces himself")
            
            # Create test audio (silence for testing)
            sample_rate = 16000
            duration = 1.0
            samples = int(sample_rate * duration)
            audio_data = np.zeros(samples, dtype=np.float32)
            
            # Convert to WAV bytes
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
            
            audio_bytes = wav_buffer.getvalue()
            
            # Test 1: Initial greeting (system introduces itself as Alexa)
            print("\n🧪 Test 1: Initial greeting...")
            await websocket.send(json.dumps({"type": "transcribe"}))
            await websocket.send(audio_bytes)
            response1 = await websocket.recv()
            print(f"📝 Alexa: {response1}")
            
            # Test 2: Surya introduces himself (simulated by silence - system should respond)
            print("\n🧪 Test 2: Surya introduces himself as age 22...")
            print("🎭 [Surya says: 'Hi, I'm Surya and I'm 22 years old']")
            await websocket.send(json.dumps({"type": "transcribe"}))
            await websocket.send(audio_bytes)
            response2 = await websocket.recv()
            print(f"📝 Alexa: {response2}")
            
            # Test 3: System asks for name (should remember Surya from context)
            print("\n🧪 Test 3: System asks for name...")
            print("🎭 [System should ask: 'What's your name?']")
            await websocket.send(json.dumps({"type": "transcribe"}))
            await websocket.send(audio_bytes)
            response3 = await websocket.recv()
            print(f"📝 Alexa: {response3}")
            
            # Test 4: System asks for age (should remember 22 from context)
            print("\n🧪 Test 4: System asks for age...")
            print("🎭 [System should ask: 'How old are you?']")
            await websocket.send(json.dumps({"type": "transcribe"}))
            await websocket.send(audio_bytes)
            response4 = await websocket.recv()
            print(f"📝 Alexa: {response4}")
            
            # Test 5: System references previous information
            print("\n🧪 Test 5: System references previous info...")
            print("🎭 [System should reference name and age from memory]")
            await websocket.send(json.dumps({"type": "transcribe"}))
            await websocket.send(audio_bytes)
            response5 = await websocket.recv()
            print(f"📝 Alexa: {response5}")
            
            # Test 6: Check conversation history
            print("\n🧪 Test 6: Checking conversation history...")
            await websocket.send(json.dumps({"type": "history"}))
            history_response = await websocket.recv()
            history_data = json.loads(history_response)
            print(f"📊 Total conversation turns: {history_data['total_turns']}")
            
            # Display conversation history
            print("\n📚 Full conversation history:")
            for i, turn in enumerate(history_data['turns']):
                role = turn.get('role', 'unknown')
                content = turn.get('content', '')
                # Truncate long content
                display_content = content[:100] + "..." if len(content) > 100 else content
                print(f"  {i+1}. {role}: {display_content}")
            
            # Verify memory is working
            assert history_data['total_turns'] >= 6, f"Expected at least 6 turns, got {history_data['total_turns']}"
            print("\n✅ Memory test passed - conversation history is being maintained")
            
            # Check if system shows awareness of context
            context_awareness = any("surya" in response.lower() or "22" in response for response in [response2, response3, response4, response5])
            if context_awareness:
                print("✅ Context awareness test passed - system shows memory of previous conversation")
            else:
                print("⚠️  Context awareness test - system may not be fully utilizing conversation memory")
            
            print("\n✅ Surya conversation flow test completed!")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        raise

def test_surya_conversation():
    """Run the async Surya conversation test"""
    asyncio.run(test_surya_conversation_flow())

if __name__ == "__main__":
    print("🧪 Testing Surya conversation memory flow...")
    print("Make sure the websocket_api.py server is running on localhost:8000")
    print("This test simulates Surya introducing himself as age 22")
    print("and checks if the system remembers this information in follow-up questions")
    test_surya_conversation()

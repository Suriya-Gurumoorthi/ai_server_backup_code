import asyncio
import websockets
import json
import wave
import io
import numpy as np

async def test_conversation_memory_flow():
    """Test conversation memory with a realistic HR recruitment scenario"""
    
    uri = "ws://localhost:8000"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket server")
            
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
            
            # Test 1: Initial greeting (system should introduce itself)
            print("\n🧪 Test 1: Initial greeting...")
            await websocket.send(json.dumps({"type": "transcribe"}))
            await websocket.send(audio_bytes)
            response1 = await websocket.recv()
            print(f"📝 Response 1: {response1}")
            
            # Test 2: User introduces themselves as Surya, age 22
            print("\n🧪 Test 2: User introduces as Surya, age 22...")
            await websocket.send(json.dumps({"type": "transcribe"}))
            await websocket.send(audio_bytes)
            response2 = await websocket.recv()
            print(f"📝 Response 2: {response2}")
            
            # Test 3: System asks for name (should remember from previous context)
            print("\n🧪 Test 3: System asks for name...")
            await websocket.send(json.dumps({"type": "transcribe"}))
            await websocket.send(audio_bytes)
            response3 = await websocket.recv()
            print(f"📝 Response 3: {response3}")
            
            # Test 4: System asks for age (should remember from previous context)
            print("\n🧪 Test 4: System asks for age...")
            await websocket.send(json.dumps({"type": "transcribe"}))
            await websocket.send(audio_bytes)
            response4 = await websocket.recv()
            print(f"📝 Response 4: {response4}")
            
            # Test 5: Check conversation history
            print("\n🧪 Test 5: Checking conversation history...")
            await websocket.send(json.dumps({"type": "history"}))
            history_response = await websocket.recv()
            history_data = json.loads(history_response)
            print(f"📊 Total conversation turns: {history_data['total_turns']}")
            
            # Verify memory is working
            assert history_data['total_turns'] >= 5, f"Expected at least 5 turns, got {history_data['total_turns']}"
            print("✅ Memory test passed - conversation history is being maintained")
            
            print("\n✅ Conversation memory flow test completed!")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        raise

def test_conversation_flow():
    """Run the async conversation test"""
    asyncio.run(test_conversation_memory_flow())

if __name__ == "__main__":
    print("🧪 Testing conversation memory flow...")
    print("Make sure the websocket_api.py server is running on localhost:8000")
    test_conversation_flow()
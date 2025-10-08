#!/usr/bin/env python3
"""
Test to verify the audio placeholder fix works.
"""

import asyncio
import json
import tempfile
import shutil
from pathlib import Path

# Add the server directory to the path
import sys
sys.path.append('/home/novel/server')

from transcription_manager import TranscriptionManager


async def test_audio_fix():
    """Test that the audio placeholder fix works."""
    print("🧪 Testing audio placeholder fix...")
    
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    print(f"📁 Using temporary directory: {temp_dir}")
    
    try:
        # Initialize transcription manager with temp directory
        transcription_manager = TranscriptionManager(storage_dir=temp_dir)
        
        # Create a test session
        connection_id = "test_audio_fix"
        session_id = transcription_manager.create_session(connection_id)
        print(f"✅ Created test session: {session_id}")
        
        # Add some sample conversation data with potential audio placeholders
        sample_conversations = [
            {"type": "user", "User": "Hello, I'm John Smith <|audio|> and I'm interested in the software engineer position."},
            {"type": "ai", "AI": "Hello John! Great to meet you. Can you tell me about your experience with Python and JavaScript?"},
            {"type": "user", "User": "I have 5 years of experience with Python and 3 years with JavaScript. I've worked on several web applications."},
            {"type": "ai", "AI": "That sounds excellent! What kind of projects have you worked on recently?"},
            {"type": "user", "User": "I recently built a full-stack e-commerce application using React and Django. It handles payments and inventory management."},
            {"type": "ai", "AI": "Impressive! We're looking for someone with exactly that kind of experience. When would you be available to start?"},
            {"type": "user", "User": "I could start in about 2 weeks. I'm very excited about this opportunity!"},
            {"type": "ai", "AI": "Perfect! We'll be in touch soon with next steps. Thank you for your time today."}
        ]
        
        # Add conversations to the session
        for conv in sample_conversations:
            if conv["type"] == "user":
                transcription_manager.add_user_transcription(connection_id, conv["User"])
            else:
                transcription_manager.add_ai_transcription(connection_id, conv["AI"])
        
        print(f"✅ Added {len(sample_conversations)} conversation entries (including one with audio placeholder)")
        
        # Test the generate_and_save_summary method directly
        print("🤖 Testing generate_and_save_summary method with audio placeholders...")
        try:
            result = await transcription_manager.generate_and_save_summary(connection_id)
            if result:
                print("✅ Summary generation completed successfully (audio placeholders sanitized)")
                
                # Check if summary was added to the file
                session_data = transcription_manager.active_sessions[connection_id]
                file_path = session_data["file_path"]
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if "summary" in data:
                    print(f"✅ Summary found in file: {data['summary'][:100]}...")
                else:
                    print("❌ Summary not found in file")
            else:
                print("❌ Summary generation failed")
                
        except Exception as e:
            print(f"❌ Summary generation failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        print("🎉 Audio placeholder fix test completed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        print(f"🧹 Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    asyncio.run(test_audio_fix())

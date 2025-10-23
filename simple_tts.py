#!/usr/bin/env python3
"""
Simple TTS script using pyttsx3 (already installed and working).
This is a replacement for the incompatible chatterbox-tts.
"""

import pyttsx3
import os

def generate_tts_audio(text: str, output_file: str = "output.wav"):
    """Generate TTS audio from text and save to file using pyttsx3."""
    try:
        # Initialize the TTS engine
        engine = pyttsx3.init()
        
        # Get available voices
        voices = engine.getProperty('voices')
        print(f"Available voices: {len(voices)}")
        
        # Set voice properties (optional)
        if voices:
            # Use the first available voice
            engine.setProperty('voice', voices[0].id)
        
        # Set speech rate (words per minute)
        engine.setProperty('rate', 150)  # Adjust as needed
        
        # Set volume (0.0 to 1.0)
        engine.setProperty('volume', 0.9)
        
        print(f"Generating TTS for: '{text}'")
        
        # Save to file
        engine.save_to_file(text, output_file)
        engine.runAndWait()
        
        # Check if file was created
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"✅ TTS audio saved to: {output_file}")
            print(f"📊 Audio size: {file_size} bytes")
            return True
        else:
            print("❌ Failed to create audio file")
            return False
            
    except Exception as e:
        print(f"❌ Error generating TTS: {e}")
        return False

def main():
    """Main function to test TTS generation."""
    # Test text (same as your original chatterbox example)
    text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
    
    print("🎤 Starting TTS generation with pyttsx3...")
    
    # Generate first audio file
    success1 = generate_tts_audio(text, "test-1.wav")
    
    if success1:
        # Generate second audio file (like your original script)
        success2 = generate_tts_audio(text, "test-2.wav")
        
        if success2:
            print("🎉 Both TTS files generated successfully!")
            print("📁 Files created:")
            print("   - test-1.wav")
            print("   - test-2.wav")
        else:
            print("❌ Failed to generate second audio file")
    else:
        print("❌ Failed to generate first audio file")

if __name__ == "__main__":
    main()



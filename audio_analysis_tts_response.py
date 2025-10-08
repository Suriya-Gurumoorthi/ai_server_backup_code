#!/usr/bin/env python3
"""
Audio Analysis and TTS Response Generator
Combines Ultravox for audio analysis and Piper TTS for response generation
"""

import os
import sys
import time
import wave
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import librosa

# Add voice_to_voice to path for Piper TTS
sys.path.append('voice_to_voice')

try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    print("⚠️  Piper TTS not available. Install with: pip install piper-tts")

# Import Ultravox components
try:
    from model_loader import get_ultravox_pipeline, is_model_loaded
    from ultravox_usage import chat_with_audio, transcribe_audio
    ULTRAVOX_AVAILABLE = True
except ImportError:
    ULTRAVOX_AVAILABLE = False
    print("⚠️  Ultravox not available. Check model_loader.py and ultravox_usage.py")

class AudioAnalysisTTSResponse:
    def __init__(self):
        """Initialize the audio analysis and TTS response system."""
        self.output_dir = Path("audio_analysis_output")
        self.output_dir.mkdir(exist_ok=True)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True)
        
        self.piper_engine = None
        self.ultravox_pipeline = None
        
        print("🎤 Audio Analysis and TTS Response System")
        print("=" * 50)
        print(f"📁 Session ID: {self.session_id}")
        print(f"📂 Output Directory: {self.session_dir}")
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize both Ultravox and Piper TTS models."""
        print("\n🔄 Initializing Models...")
        
        # Initialize Ultravox
        if ULTRAVOX_AVAILABLE:
            print("📦 Loading Ultravox model...")
            self.ultravox_pipeline = get_ultravox_pipeline()
            if self.ultravox_pipeline:
                print("✅ Ultravox model loaded successfully!")
            else:
                print("❌ Failed to load Ultravox model")
        else:
            print("❌ Ultravox not available")
        
        # Initialize Piper TTS
        if PIPER_AVAILABLE:
            print("🎵 Loading Piper TTS...")
            model_paths = self._find_piper_models()
            if model_paths:
                try:
                    self.piper_engine = PiperVoice.load(model_paths['model'], model_paths['config'])
                    print("✅ Piper TTS loaded successfully!")
                except Exception as e:
                    print(f"❌ Failed to load Piper TTS: {e}")
            else:
                print("❌ Piper models not found!")
        else:
            print("❌ Piper TTS not available")
    
    def _find_piper_models(self):
        """Find Piper TTS models in common locations."""
        search_paths = [
            "voice_to_voice/models",
            "pretrained_models", 
            "~/.local/share/piper/voices",
            "/usr/local/share/piper/voices"
        ]
        
        for path in search_paths:
            expanded_path = Path(path).expanduser()
            if expanded_path.exists():
                onnx_files = list(expanded_path.glob("*.onnx"))
                json_files = list(expanded_path.glob("*.json"))
                
                if onnx_files and json_files:
                    return {
                        'model': str(onnx_files[0]),
                        'config': str(json_files[0])
                    }
        
        return None
    
    def analyze_audio(self, audio_file_path):
        """
        Analyze audio using Ultravox model.
        
        Args:
            audio_file_path (str): Path to the input audio file
            
        Returns:
            dict: Analysis results including transcription and response
        """
        if not self.ultravox_pipeline:
            return {"error": "Ultravox model not available"}
        
        if not os.path.exists(audio_file_path):
            return {"error": f"Audio file not found: {audio_file_path}"}
        
        print(f"\n🎤 Analyzing audio: {audio_file_path}")
        print("-" * 40)
        
        # Load and validate audio
        try:
            audio, sr = librosa.load(audio_file_path, sr=16000)
            duration = len(audio) / sr
            print(f"📊 Audio loaded: {duration:.2f}s duration, {sr}Hz sample rate")
        except Exception as e:
            return {"error": f"Failed to load audio: {e}"}
        
        # Transcribe audio
        print("🔄 Transcribing audio...")
        transcription_result = transcribe_audio(audio_file_path)
        
        if "error" in transcription_result:
            return transcription_result
        
        transcription = transcription_result.get("transcription", "")
        print(f"📝 Transcription: {transcription}")
        
        # Generate response using Ultravox
        print("🔄 Generating response...")
        response_result = chat_with_audio(
            audio_file_path=audio_file_path,
            user_message="Please analyze this audio and provide a helpful response.",
            system_prompt="You are a helpful AI assistant. Analyze the audio input and provide a relevant, helpful response. Keep responses concise and natural.",
            max_tokens=100
        )
        
        if "error" in response_result:
            return response_result
        
        response_text = response_result.get("response", "")
        if isinstance(response_text, dict):
            response_text = response_text.get("generated_text", "")
        
        print(f"💬 Response: {response_text}")
        
        return {
            "transcription": transcription,
            "response": response_text,
            "audio_duration": duration,
            "sample_rate": sr,
            "analysis_time": time.time()
        }
    
    def generate_tts_response(self, text, filename=None):
        """
        Generate TTS audio from text using Piper TTS.
        
        Args:
            text (str): Text to convert to speech
            filename (str): Optional filename for output
            
        Returns:
            str: Path to generated audio file
        """
        if not self.piper_engine:
            return None
        
        if not filename:
            timestamp = datetime.now().strftime("%H%M%S")
            filename = f"tts_response_{timestamp}.wav"
        
        output_path = self.session_dir / filename
        
        try:
            print(f"🎵 Generating TTS audio: {filename}")
            with wave.open(str(output_path), "wb") as wav_file:
                self.piper_engine.synthesize_wav(text, wav_file)
            
            print(f"✅ TTS audio generated: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ TTS generation error: {e}")
            return None
    
    def process_audio_file(self, audio_file_path):
        """
        Complete workflow: analyze audio and generate TTS response.
        
        Args:
            audio_file_path (str): Path to input audio file
            
        Returns:
            dict: Complete processing results
        """
        print(f"\n🚀 Processing Audio File: {audio_file_path}")
        print("=" * 60)
        
        # Validate input
        if not os.path.exists(audio_file_path):
            return {"error": f"Audio file not found: {audio_file_path}"}
        
        # Create session log
        log_file = self.session_dir / "processing_log.txt"
        with open(log_file, "w") as f:
            f.write(f"Audio Analysis and TTS Response Session\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Input File: {audio_file_path}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 50 + "\n\n")
        
        # Step 1: Analyze audio
        print("📊 Step 1: Audio Analysis")
        analysis_result = self.analyze_audio(audio_file_path)
        
        if "error" in analysis_result:
            print(f"❌ Analysis failed: {analysis_result['error']}")
            return analysis_result
        
        # Step 2: Generate TTS response
        print("\n🎵 Step 2: TTS Response Generation")
        response_text = analysis_result.get("response", "")
        
        if not response_text:
            response_text = "I'm sorry, I couldn't generate a response for that audio input."
        
        tts_file = self.generate_tts_response(response_text, "ai_response.wav")
        
        # Step 3: Save results
        print("\n💾 Step 3: Saving Results")
        
        # Save analysis results
        results_file = self.session_dir / "analysis_results.json"
        results = {
            "session_id": self.session_id,
            "input_audio": audio_file_path,
            "analysis": analysis_result,
            "tts_response_file": tts_file,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Update log
        with open(log_file, "a") as f:
            f.write(f"Analysis Results:\n")
            f.write(f"Transcription: {analysis_result.get('transcription', '')}\n")
            f.write(f"Response: {analysis_result.get('response', '')}\n")
            f.write(f"TTS Output: {tts_file}\n")
            f.write(f"Results saved to: {results_file}\n")
        
        print(f"✅ Results saved to: {results_file}")
        print(f"📝 Processing log: {log_file}")
        
        # Final summary
        print("\n🎉 Processing Complete!")
        print("=" * 40)
        print(f"📁 Session Directory: {self.session_dir}")
        print(f"🎵 TTS Response: {tts_file}")
        print(f"📊 Analysis Results: {results_file}")
        print(f"📝 Processing Log: {log_file}")
        
        return results
    
    def list_sessions(self):
        """List all processing sessions."""
        sessions = list(self.output_dir.glob("session_*"))
        if not sessions:
            print("No sessions found.")
            return
        
        print("📋 Available Sessions:")
        for session in sorted(sessions, reverse=True):
            session_id = session.name
            log_file = session / "processing_log.txt"
            if log_file.exists():
                with open(log_file, "r") as f:
                    first_line = f.readline().strip()
                print(f"  {session_id}: {first_line}")
            else:
                print(f"  {session_id}")

def main():
    """Main function to demonstrate the audio analysis and TTS response system."""
    print("🎤 Audio Analysis and TTS Response Demo")
    print("=" * 50)
    
    # Initialize the system
    processor = AudioAnalysisTTSResponse()
    
    # Check if models are available
    if not processor.ultravox_pipeline:
        print("❌ Ultravox model not available. Cannot proceed with audio analysis.")
        return
    
    if not processor.piper_engine:
        print("❌ Piper TTS not available. Cannot generate TTS responses.")
        return
    
    # Get input audio file
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        # Look for test audio files
        test_files = [
            "voice_to_voice/custom_test.wav",
            "voice_to_voice/test_custom.wav", 
            "voice_to_voice/test.wav",
            "temp_audio/test.wav"
        ]
        
        audio_file = None
        for test_file in test_files:
            if os.path.exists(test_file):
                audio_file = test_file
                break
        
        if not audio_file:
            print("❌ No audio file provided and no test files found.")
            print("Usage: python audio_analysis_tts_response.py <audio_file_path>")
            print("\nOr place a test audio file in one of these locations:")
            for test_file in test_files:
                print(f"  - {test_file}")
            return
    
    # Process the audio file
    results = processor.process_audio_file(audio_file)
    
    if "error" not in results:
        print("\n🎵 To play the generated TTS response:")
        print(f"   aplay {results['tts_response_file']}")
        print(f"   or open: {results['tts_response_file']}")
    
    # List all sessions
    print("\n📋 All Sessions:")
    processor.list_sessions()

if __name__ == "__main__":
    main()


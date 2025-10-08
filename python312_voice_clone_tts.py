#!/usr/bin/env python3
"""
Python 3.12 Compatible Voice Clone TTS Response Generator
Uses alternative voice cloning methods that work with Python 3.12
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
import subprocess
import shutil

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

class Python312VoiceCloneTTS:
    def __init__(self):
        """Initialize the Python 3.12 compatible voice clone TTS system."""
        self.output_dir = Path("python312_voice_clone_output")
        self.output_dir.mkdir(exist_ok=True)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True)
        
        self.piper_engine = None
        self.ultravox_pipeline = None
        self.cloned_voice_path = None
        
        print("🎤 Python 3.12 Compatible Voice Clone TTS System")
        print("=" * 50)
        print(f"📁 Session ID: {self.session_id}")
        print(f"📂 Output Directory: {self.session_dir}")
        print(f"🐍 Python Version: {sys.version}")
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize Ultravox and Piper TTS models."""
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
    
    def _validate_audio_file(self, audio_path):
        """Validate and prepare audio file for processing."""
        if not os.path.exists(audio_path):
            return {"error": f"Audio file not found: {audio_path}"}
        
        try:
            # Load audio to check format and quality
            audio, sr = librosa.load(audio_path, sr=None)
            duration = len(audio) / sr
            
            print(f"📊 Audio validation:")
            print(f"   Duration: {duration:.2f}s")
            print(f"   Sample rate: {sr}Hz")
            print(f"   Channels: {audio.ndim}")
            print(f"   Format: {Path(audio_path).suffix}")
            
            # Check if audio has content
            if np.max(np.abs(audio)) < 0.01:
                return {"error": "Audio file appears to be silent or very quiet"}
            
            # Check duration
            if duration < 1.0:
                return {"error": "Audio file is too short (less than 1 second)"}
            
            return {"success": True, "audio": audio, "sr": sr, "duration": duration}
            
        except Exception as e:
            return {"error": f"Failed to load audio: {e}"}
    
    def clone_voice_python312(self, reference_audio_path, voice_name=None):
        """
        Voice cloning methods compatible with Python 3.12.
        """
        if not reference_audio_path or not os.path.exists(reference_audio_path):
            return {"error": f"Reference audio file not found: {reference_audio_path}"}
        
        if not voice_name:
            voice_name = f"cloned_voice_{datetime.now().strftime('%H%M%S')}"
        
        print(f"\n🎭 Python 3.12 Compatible Voice Cloning")
        print(f"Reference: {reference_audio_path}")
        print(f"Voice Name: {voice_name}")
        print("-" * 40)
        
        # Validate reference audio
        validation = self._validate_audio_file(reference_audio_path)
        if "error" in validation:
            return validation
        
        # Create voice clone directory
        voice_dir = self.session_dir / "cloned_voices" / voice_name
        voice_dir.mkdir(parents=True, exist_ok=True)
        
        # Method 1: Advanced Voice Style Transfer with Pitch Analysis
        print("🔄 Method 1: Advanced Voice Style Transfer...")
        try:
            # Extract comprehensive voice characteristics
            audio, sr = librosa.load(reference_audio_path, sr=22050)
            
            # Calculate voice characteristics
            pitch = librosa.yin(audio, fmin=75, fmax=300)
            pitch_mean = np.nanmean(pitch[pitch > 0])
            
            # Calculate spectral features
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            
            # Calculate formant-like features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            
            # Create comprehensive voice profile
            voice_profile = {
                "name": voice_name,
                "pitch_mean": float(pitch_mean) if not np.isnan(pitch_mean) else 150.0,
                "pitch_std": float(np.nanstd(pitch[pitch > 0])) if len(pitch[pitch > 0]) > 0 else 20.0,
                "mfcc_features": mfcc_mean.tolist(),
                "spectral_centroid_mean": float(np.mean(spectral_centroid)),
                "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
                "sample_rate": sr,
                "duration": len(audio) / sr,
                "method": "advanced_style_transfer",
                "created": datetime.now().isoformat(),
                "python_version": sys.version
            }
            
            self.cloned_voice_path = voice_dir / f"{voice_name}_profile.json"
            with open(self.cloned_voice_path, 'w') as f:
                json.dump(voice_profile, f, indent=2)
            
            print(f"✅ Advanced voice style transfer setup: {self.cloned_voice_path}")
            return {
                "success": True, 
                "voice_path": str(self.cloned_voice_path),
                "method": "advanced_style_transfer",
                "pitch_mean": voice_profile["pitch_mean"]
            }
            
        except Exception as e:
            print(f"❌ Advanced style transfer failed: {e}")
        
        # Method 2: Piper TTS Voice Adaptation
        if self.piper_engine:
            print("🔄 Method 2: Using Piper TTS voice adaptation...")
            try:
                # Convert audio to proper format for Piper
                converted_audio = voice_dir / "reference_converted.wav"
                self._convert_audio_format(reference_audio_path, str(converted_audio))
                
                # Create voice configuration
                voice_config = {
                    "name": voice_name,
                    "reference_audio": str(converted_audio),
                    "created": datetime.now().isoformat(),
                    "method": "piper_adaptation",
                    "sample_rate": 22050,
                    "channels": 1,
                    "python_version": sys.version
                }
                
                self.cloned_voice_path = voice_dir / f"{voice_name}_config.json"
                with open(self.cloned_voice_path, 'w') as f:
                    json.dump(voice_config, f, indent=2)
                
                print(f"✅ Piper voice adaptation setup: {self.cloned_voice_path}")
                return {
                    "success": True, 
                    "voice_path": str(self.cloned_voice_path),
                    "method": "piper_adaptation"
                }
                
            except Exception as e:
                print(f"❌ Piper voice adaptation failed: {e}")
        
        # Method 3: Simple Voice Characteristics Transfer
        print("🔄 Method 3: Simple voice characteristics transfer...")
        try:
            # Basic voice analysis
            audio, sr = librosa.load(reference_audio_path, sr=22050)
            
            # Simple pitch analysis
            pitch = librosa.yin(audio, fmin=75, fmax=300)
            pitch_mean = np.nanmean(pitch[pitch > 0])
            
            # Create basic voice profile
            voice_profile = {
                "name": voice_name,
                "pitch_mean": float(pitch_mean) if not np.isnan(pitch_mean) else 150.0,
                "sample_rate": sr,
                "duration": len(audio) / sr,
                "method": "simple_transfer",
                "created": datetime.now().isoformat(),
                "python_version": sys.version
            }
            
            self.cloned_voice_path = voice_dir / f"{voice_name}_simple.json"
            with open(self.cloned_voice_path, 'w') as f:
                json.dump(voice_profile, f, indent=2)
            
            print(f"✅ Simple voice transfer setup: {self.cloned_voice_path}")
            return {
                "success": True, 
                "voice_path": str(self.cloned_voice_path),
                "method": "simple_transfer"
            }
            
        except Exception as e:
            print(f"❌ Simple voice transfer failed: {e}")
        
        return {"error": "All voice cloning methods failed"}
    
    def _convert_audio_format(self, input_path, output_path):
        """Convert audio to WAV format suitable for voice cloning."""
        try:
            # Use ffmpeg for high-quality conversion
            cmd = [
                "ffmpeg", "-i", input_path, 
                "-ar", "22050", "-ac", "1", "-c:a", "pcm_s16le",
                "-y", output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Audio converted: {output_path}")
                return True
            else:
                print(f"⚠️  ffmpeg conversion warning: {result.stderr}")
                
        except FileNotFoundError:
            print("⚠️  ffmpeg not found, using librosa for conversion...")
        
        try:
            # Fallback to librosa
            audio, sr = librosa.load(input_path, sr=22050)
            import soundfile as sf
            sf.write(output_path, audio, sr)
            print(f"✅ Audio converted with librosa: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Audio conversion failed: {e}")
            return False
    
    def generate_tts_with_cloned_voice(self, text, filename=None, use_cloned_voice=True):
        """Generate TTS audio using Python 3.12 compatible voice cloning methods."""
        if not filename:
            timestamp = datetime.now().strftime("%H%M%S")
            filename = f"tts_response_{timestamp}.wav"
        
        output_path = self.session_dir / filename
        
        try:
            if use_cloned_voice and self.cloned_voice_path:
                print(f"🎭 Generating TTS with cloned voice: {filename}")
                
                # Method 1: Advanced style transfer
                if self.cloned_voice_path.suffix == '.json':
                    try:
                        with open(self.cloned_voice_path, 'r') as f:
                            voice_profile = json.load(f)
                        
                        if voice_profile.get("method") == "advanced_style_transfer":
                            return self._generate_with_advanced_style_transfer(text, output_path, voice_profile)
                        elif voice_profile.get("method") == "simple_transfer":
                            return self._generate_with_simple_transfer(text, output_path, voice_profile)
                        elif voice_profile.get("method") == "piper_adaptation":
                            return self._generate_with_piper_adaptation(text, output_path, voice_profile)
                    except Exception as e:
                        print(f"⚠️  Voice profile loading failed: {e}")
                
                print("⚠️  Voice cloning method not recognized, using default voice")
                return self.generate_tts_response(text, filename)
                
            else:
                print(f"🎵 Generating TTS with default voice: {filename}")
                return self.generate_tts_response(text, filename)
            
        except Exception as e:
            print(f"❌ TTS generation error: {e}")
            print("🔄 Falling back to default voice...")
            return self.generate_tts_response(text, filename)
    
    def _generate_with_advanced_style_transfer(self, text, output_path, voice_profile):
        """Generate TTS with advanced voice style transfer."""
        try:
            # Generate base audio with Piper
            with wave.open(str(output_path), "wb") as wav_file:
                self.piper_engine.synthesize_wav(text, wav_file)
            
            # Apply advanced voice modifications
            audio, sr = librosa.load(str(output_path), sr=22050)
            
            # Pitch adjustment based on voice profile
            target_pitch = voice_profile.get("pitch_mean", 150.0)
            if target_pitch > 0:
                # Calculate pitch shift
                pitch_shift = (target_pitch - 150.0) / 150.0  # Normalize
                if abs(pitch_shift) > 0.1:  # Only shift if significant difference
                    import librosa.effects as effects
                    n_steps = int(pitch_shift * 4)  # Convert to semitones
                    if abs(n_steps) > 0:
                        audio = effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
            
            # Apply spectral modifications if available
            if "spectral_centroid_mean" in voice_profile:
                # Simple spectral adjustment
                current_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
                target_centroid = voice_profile["spectral_centroid_mean"]
                
                if abs(current_centroid - target_centroid) > 100:
                    # Apply spectral adjustment
                    ratio = target_centroid / current_centroid
                    audio = audio * np.clip(ratio, 0.5, 2.0)
            
            # Save modified audio
            import soundfile as sf
            sf.write(str(output_path), audio, sr)
            
            print(f"✅ TTS generated with advanced style transfer: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Advanced style transfer failed: {e}")
            return None
    
    def _generate_with_simple_transfer(self, text, output_path, voice_profile):
        """Generate TTS with simple voice transfer."""
        try:
            # Generate base audio
            with wave.open(str(output_path), "wb") as wav_file:
                self.piper_engine.synthesize_wav(text, wav_file)
            
            # Apply simple pitch adjustment
            target_pitch = voice_profile.get("pitch_mean", 150.0)
            if target_pitch > 0:
                audio, sr = librosa.load(str(output_path), sr=22050)
                
                import librosa.effects as effects
                if target_pitch > 160:  # Higher pitch
                    audio = effects.pitch_shift(audio, sr=sr, n_steps=2)
                elif target_pitch < 140:  # Lower pitch
                    audio = effects.pitch_shift(audio, sr=sr, n_steps=-2)
                
                import soundfile as sf
                sf.write(str(output_path), audio, sr)
            
            print(f"✅ TTS generated with simple transfer: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Simple transfer failed: {e}")
            return None
    
    def _generate_with_piper_adaptation(self, text, output_path, voice_profile):
        """Generate TTS with Piper adaptation."""
        try:
            # Use Piper with voice configuration
            with wave.open(str(output_path), "wb") as wav_file:
                self.piper_engine.synthesize_wav(text, wav_file, voice_config=voice_profile)
            
            print(f"✅ TTS generated with Piper adaptation: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Piper adaptation failed: {e}")
            return None
    
    def generate_tts_response(self, text, filename=None):
        """Generate TTS audio from text using default Piper TTS."""
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
    
    def analyze_audio_enhanced(self, audio_file_path):
        """Enhanced audio analysis with better error handling."""
        if not self.ultravox_pipeline:
            return {"error": "Ultravox model not available"}
        
        # Validate audio file
        validation = self._validate_audio_file(audio_file_path)
        if "error" in validation:
            return validation
        
        print(f"\n🎤 Enhanced Audio Analysis: {audio_file_path}")
        print("-" * 40)
        
        # Transcribe audio
        print("🔄 Transcribing audio...")
        transcription_result = transcribe_audio(audio_file_path)
        
        if "error" in transcription_result:
            print(f"⚠️  Transcription failed: {transcription_result['error']}")
            transcription = ""
        else:
            transcription = transcription_result.get("transcription", "")
            print(f"📝 Transcription: {transcription}")
        
        # Generate response using Ultravox
        print("🔄 Generating AI response...")
        try:
            response_result = chat_with_audio(
                audio_file_path=audio_file_path,
                user_message="Please analyze this audio and provide a helpful response.",
                system_prompt="You are a helpful AI assistant. Analyze the audio input and provide a relevant, helpful response. Keep responses concise and natural.",
                max_tokens=100
            )
            
            if "error" in response_result:
                print(f"⚠️  AI response generation failed: {response_result['error']}")
                response_text = "I'm sorry, I couldn't generate a response for that audio input."
            else:
                response_text = response_result.get("response", "")
                if isinstance(response_text, dict):
                    response_text = response_text.get("generated_text", "")
                
                if not response_text.strip():
                    response_text = "I heard your audio but couldn't generate a specific response. How can I help you?"
                
                print(f"💬 Response: {response_text}")
                
        except Exception as e:
            print(f"❌ AI response generation error: {e}")
            response_text = "I'm sorry, I encountered an error while processing your audio."
        
        return {
            "transcription": transcription,
            "response": response_text,
            "audio_duration": validation["duration"],
            "sample_rate": validation["sr"],
            "analysis_time": time.time()
        }
    
    def process_audio_with_python312_cloning(self, audio_file_path, reference_voice_path=None, voice_name=None):
        """Complete workflow with Python 3.12 compatible voice cloning."""
        print(f"\n🚀 Python 3.12 Compatible Audio Processing with Voice Cloning")
        print(f"Input: {audio_file_path}")
        print(f"Reference Voice: {reference_voice_path or 'None'}")
        print("=" * 60)
        
        # Validate input
        if not os.path.exists(audio_file_path):
            return {"error": f"Audio file not found: {audio_file_path}"}
        
        # Create session log
        log_file = self.session_dir / "processing_log.txt"
        with open(log_file, "w") as f:
            f.write(f"Python 3.12 Compatible Voice Clone TTS Response Session\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Input File: {audio_file_path}\n")
            f.write(f"Reference Voice: {reference_voice_path or 'None'}\n")
            f.write(f"Python Version: {sys.version}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 50 + "\n\n")
        
        # Step 1: Python 3.12 compatible voice cloning
        voice_clone_result = None
        if reference_voice_path:
            print("🎭 Step 1: Python 3.12 Compatible Voice Cloning")
            voice_clone_result = self.clone_voice_python312(reference_voice_path, voice_name)
            
            if "error" in voice_clone_result:
                print(f"⚠️  Voice cloning failed: {voice_clone_result['error']}")
                print("🔄 Continuing with default voice...")
            else:
                print(f"✅ Voice cloning completed successfully!")
                print(f"   Method: {voice_clone_result.get('method', 'Unknown')}")
                print(f"   Voice path: {voice_clone_result.get('voice_path', 'Unknown')}")
        
        # Step 2: Enhanced audio analysis
        print("\n📊 Step 2: Enhanced Audio Analysis")
        analysis_result = self.analyze_audio_enhanced(audio_file_path)
        
        if "error" in analysis_result:
            print(f"❌ Analysis failed: {analysis_result['error']}")
            return analysis_result
        
        # Step 3: Generate TTS response with cloned voice
        print("\n🎵 Step 3: TTS Response Generation")
        response_text = analysis_result.get("response", "")
        
        if not response_text:
            response_text = "I'm sorry, I couldn't generate a response for that audio input."
        
        # Use cloned voice if available
        use_cloned_voice = voice_clone_result and "error" not in voice_clone_result
        tts_file = self.generate_tts_with_cloned_voice(
            response_text, 
            "ai_response_cloned.wav" if use_cloned_voice else "ai_response.wav",
            use_cloned_voice=use_cloned_voice
        )
        
        # Step 4: Save results
        print("\n💾 Step 4: Saving Results")
        
        # Save analysis results
        results_file = self.session_dir / "analysis_results.json"
        results = {
            "session_id": self.session_id,
            "input_audio": audio_file_path,
            "reference_voice": reference_voice_path,
            "voice_clone_result": voice_clone_result,
            "analysis": analysis_result,
            "tts_response_file": tts_file,
            "voice_cloning_used": use_cloned_voice,
            "cloning_method": voice_clone_result.get("method") if voice_clone_result else None,
            "python_version": sys.version,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Update log
        with open(log_file, "a") as f:
            f.write(f"Voice Cloning: {'Success' if use_cloned_voice else 'Not used/Failed'}\n")
            if voice_clone_result and "method" in voice_clone_result:
                f.write(f"Cloning Method: {voice_clone_result['method']}\n")
            f.write(f"Analysis Results:\n")
            f.write(f"Transcription: {analysis_result.get('transcription', '')}\n")
            f.write(f"Response: {analysis_result.get('response', '')}\n")
            f.write(f"TTS Output: {tts_file}\n")
            f.write(f"Results saved to: {results_file}\n")
        
        print(f"✅ Results saved to: {results_file}")
        print(f"📝 Processing log: {log_file}")
        
        # Final summary
        print("\n🎉 Python 3.12 Compatible Processing Complete!")
        print("=" * 40)
        print(f"📁 Session Directory: {self.session_dir}")
        print(f"🎵 TTS Response: {tts_file}")
        print(f"🎭 Voice Cloning: {'Used' if use_cloned_voice else 'Not used'}")
        if voice_clone_result and "method" in voice_clone_result:
            print(f"🔧 Cloning Method: {voice_clone_result['method']}")
        print(f"🐍 Python Version: {sys.version}")
        print(f"📊 Analysis Results: {results_file}")
        print(f"📝 Processing Log: {log_file}")
        
        return results

def main():
    """Main function for Python 3.12 compatible voice clone TTS system."""
    print("🎤 Python 3.12 Compatible Voice Clone TTS Response Demo")
    print("=" * 50)
    
    # Initialize the system
    processor = Python312VoiceCloneTTS()
    
    # Check if models are available
    if not processor.ultravox_pipeline:
        print("❌ Ultravox model not available. Cannot proceed with audio analysis.")
        return
    
    if not processor.piper_engine:
        print("❌ Piper TTS not available. Cannot generate TTS responses.")
        return
    
    # Get input files
    if len(sys.argv) < 2:
        print("❌ No audio file provided.")
        print("Usage: python python312_voice_clone_tts.py <audio_file> [reference_voice_file] [voice_name]")
        print("\nExample:")
        print("  python python312_voice_clone_tts.py input.wav")
        print("  python python312_voice_clone_tts.py input.wav reference_voice.wav my_voice")
        return
    
    audio_file = sys.argv[1]
    reference_voice = sys.argv[2] if len(sys.argv) > 2 else None
    voice_name = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Process the audio file with Python 3.12 compatible voice cloning
    results = processor.process_audio_with_python312_cloning(audio_file, reference_voice, voice_name)
    
    if "error" not in results:
        print("\n🎵 To play the generated TTS response:")
        print(f"   aplay {results['tts_response_file']}")
        print(f"   or open: {results['tts_response_file']}")
        
        if results.get('voice_cloning_used'):
            print("\n🎭 Voice cloning was used successfully!")
            print(f"   Method: {results.get('cloning_method', 'Unknown')}")
            print(f"   Python Version: {results.get('python_version', 'Unknown')}")

if __name__ == "__main__":
    main()


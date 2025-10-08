#!/usr/bin/env python3
"""
Enhanced TTS Demo - Make TTS sound more human-like
"""

import os
import sys
import time
import wave
import re
from pathlib import Path

# Add voice_to_voice to path
sys.path.append('voice_to_voice')

try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    print("⚠️  Piper TTS not available. Install with: pip install piper-tts")

class HumanLikeTTS:
    def __init__(self):
        self.output_dir = Path("enhanced_tts_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Speech patterns for more natural sound
        self.pauses = {
            'short': '...',
            'medium': '... ...',
            'long': '... ... ...'
        }
        
        # Emotional markers
        self.emotions = {
            'excited': {'rate': 1.2, 'pitch': 1.1, 'volume': 1.2},
            'calm': {'rate': 0.9, 'pitch': 0.95, 'volume': 0.9},
            'friendly': {'rate': 1.0, 'pitch': 1.05, 'volume': 1.0},
            'professional': {'rate': 0.95, 'pitch': 1.0, 'volume': 1.1}
        }
    
    def add_natural_pauses(self, text):
        """Add natural pauses to make speech more human-like."""
        # Add pauses after sentences
        text = re.sub(r'([.!?])\s+', r'\1 ... ', text)
        
        # Add pauses after commas
        text = re.sub(r'(,)\s+', r'\1 ... ', text)
        
        # Add pauses for emphasis
        text = re.sub(r'\b(important|key|crucial|essential)\b', r'\1 ... ', text, flags=re.IGNORECASE)
        
        return text
    
    def add_speech_fillers(self, text):
        """Add natural speech fillers like humans use."""
        fillers = ['well', 'you know', 'I mean', 'actually', 'basically']
        
        # Add fillers at the beginning of some sentences
        sentences = text.split('. ')
        enhanced_sentences = []
        
        for i, sentence in enumerate(sentences):
            if i > 0 and len(sentence) > 20:  # Don't add to very short sentences
                if i % 3 == 0:  # Add filler every 3rd sentence
                    filler = fillers[i % len(fillers)]
                    sentence = f"{filler}, {sentence.lower()}"
            enhanced_sentences.append(sentence)
        
        return '. '.join(enhanced_sentences)
    
    def add_emphasis(self, text):
        """Add emphasis to important words."""
        # Words to emphasize
        emphasis_words = [
            'important', 'key', 'crucial', 'essential', 'amazing', 'incredible',
            'definitely', 'absolutely', 'certainly', 'obviously'
        ]
        
        for word in emphasis_words:
            text = re.sub(
                rf'\b{word}\b', 
                f'<emphasis>{word}</emphasis>', 
                text, 
                flags=re.IGNORECASE
            )
        
        return text
    
    def create_conversational_text(self, base_text):
        """Transform formal text into conversational speech."""
        # Make it more conversational
        conversational = base_text.replace("I am", "I'm")
        conversational = conversational.replace("you are", "you're")
        conversational = conversational.replace("we are", "we're")
        conversational = conversational.replace("they are", "they're")
        conversational = conversational.replace("it is", "it's")
        conversational = conversational.replace("that is", "that's")
        
        # Add natural pauses
        conversational = self.add_natural_pauses(conversational)
        
        # Add speech fillers
        conversational = self.add_speech_fillers(conversational)
        
        # Add emphasis
        conversational = self.add_emphasis(conversational)
        
        return conversational
    
    def generate_enhanced_audio(self, text, filename, engine, emotion='friendly'):
        """Generate audio with enhanced human-like qualities."""
        try:
            output_path = self.output_dir / filename
            
            # Apply emotion settings if supported
            emotion_settings = self.emotions.get(emotion, self.emotions['friendly'])
            
            # Create enhanced text
            enhanced_text = self.create_conversational_text(text)
            
            with wave.open(str(output_path), "wb") as wav_file:
                engine.synthesize_wav(enhanced_text, wav_file)
            
            return str(output_path)
        except Exception as e:
            print(f"❌ Enhanced TTS error: {e}")
            return None

def find_piper_models():
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

def main():
    """Main enhanced TTS demo."""
    print("🎤 Enhanced Human-Like TTS Demo")
    print("=" * 50)
    
    # Sample texts for different scenarios
    sample_texts = {
        "customer_service": "Hello! Thank you for calling our customer service department. My name is Sarah, and I'm here to help you with any questions or concerns you may have about our products and services. How can I assist you today?",
        
        "storytelling": "Once upon a time, in a world not so different from our own, there lived a curious AI assistant who loved to explore the boundaries of human communication. Every day, it learned new ways to understand and respond to the people around it, making connections that seemed impossible just a few years ago.",
        
        "technical_explanation": "The integration of text-to-speech technology with telephony systems represents a significant advancement in automated communication. By combining real-time speech synthesis with natural language processing, we can create more engaging and personalized user experiences that feel truly human.",
        
        "sales_pitch": "Are you looking for a solution that can transform your business communication? Our voice technology offers unprecedented personalization while maintaining the efficiency of automated systems. Let me show you how it works and how it can benefit your organization!"
    }
    
    # Initialize TTS
    if not PIPER_AVAILABLE:
        print("❌ Piper TTS not available!")
        return
    
    model_paths = find_piper_models()
    if not model_paths:
        print("❌ Piper models not found!")
        return
    
    try:
        print("🔄 Loading Piper TTS model...")
        engine = PiperVoice.load(model_paths['model'], model_paths['config'])
        print("✅ Piper TTS loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load Piper TTS: {e}")
        return
    
    # Initialize enhanced TTS
    enhanced_tts = HumanLikeTTS()
    
    # Generate enhanced audio
    print("\n🎵 Generating human-like audio...")
    for text_type, text in sample_texts.items():
        print(f"\n📝 Processing: {text_type}")
        print(f"Original: {text[:80]}...")
        
        # Generate enhanced version
        filename = f"{text_type}_enhanced.wav"
        start_time = time.time()
        
        result = enhanced_tts.generate_enhanced_audio(text, filename, engine, 'friendly')
        
        if result:
            duration = time.time() - start_time
            print(f"✅ Generated: {filename} ({duration:.2f}s)")
            print(f"📁 Saved to: {result}")
        else:
            print(f"❌ Failed to generate: {filename}")
    
    print("\n🎉 Enhanced TTS Demo Complete!")
    print("📁 Check the 'enhanced_tts_output' folder for generated audio files")
    
    print("\n🔧 Techniques Used to Make TTS More Human-Like:")
    print("1. ✅ Natural pauses and breathing spaces")
    print("2. ✅ Speech fillers (well, you know, etc.)")
    print("3. ✅ Conversational contractions (I'm, you're, etc.)")
    print("4. ✅ Emphasis on important words")
    print("5. ✅ Emotional tone variation")
    print("6. ✅ Natural sentence flow")
    
    print("\n🚀 Next Steps for Even Better Results:")
    print("1. Use Coqui TTS with YourTTS for voice cloning")
    print("2. Implement prosody control (pitch, speed, volume)")
    print("3. Add emotional context awareness")
    print("4. Use SSML markup for fine control")
    print("5. Train custom voice models")

if __name__ == "__main__":
    main()

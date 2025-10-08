# pip3 install transformers peft librosa huggingface_hub

# hf-auth-login
import transformers
import numpy as np
import librosa
from huggingface_hub import login
import torch

# Check CUDA availability
print("=== CUDA AVAILABILITY CHECK ===")
if torch.cuda.is_available():
    print(f"✅ CUDA is available!")
    print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    device = "cuda"
else:
    print("❌ CUDA is not available. Using CPU.")
    device = "cpu"

print(f"   Using device: {device}")
print("=" * 40)

# Login to Hugging Face (you'll be prompted for your token)
# You can get your token from: https://huggingface.co/settings/tokens
# login()

# Initialize pipeline with device configuration
print(f"\nInitializing Ultravox model on {device}...")
pipe = transformers.pipeline(
    model='fixie-ai/ultravox-v0_5-llama-3_2-1b', 
    trust_remote_code=True,
    device=device
)

path = "sample-pack-links-in-bio-sampled-stuff-288267.mp3"  # TODO: pass the audio here
audio, sr = librosa.load(path, sr=16000)

print(f"Audio loaded: {len(audio)} samples at {sr}Hz sample rate")
print(f"Audio duration: {len(audio)/sr:.2f} seconds")

turns = [
  {
    "role": "system",
    "content": "You are an expert audio transcriptionist. Your task is to provide a word-for-word, accurate transcript of the audio content. Include all words, punctuation, numbers, and special characters exactly as they appear in the audio. Do not add any commentary, interpretation, or additional text. Provide only the pure transcript."
  },
]

# Process the audio and get the response
print(f"\nProcessing audio with Ultravox model on {device} for transcription...")
response = pipe({'audio': audio, 'turns': turns, 'sampling_rate': sr}, max_new_tokens=1000)

print("\n=== TRANSCRIPTION RESULT ===")
print("=" * 50)

# Extract and display the transcript
if 'generated_text' in response:
    transcript = response['generated_text']
    print("TRANSCRIPT:")
    print(transcript)
elif 'text' in response:
    transcript = response['text']
    print("TRANSCRIPT:")
    print(transcript)
else:
    print("Full response structure:")
    print(f"Response type: {type(response)}")
    if hasattr(response, 'keys'):
        print(f"Available keys: {list(response.keys())}")
    print(f"Raw response: {response}")

print("\n" + "=" * 50)
print("Transcription complete!")

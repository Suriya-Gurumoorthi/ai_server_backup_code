import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# Load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Basic transcription function
def transcribe_audio(audio_path, chunk_length=30):
    # For local audio file
    import librosa
    import numpy as np
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)  # Whisper requires 16kHz
    
    # Calculate total duration
    duration = len(audio) / sr
    print(f"Audio duration: {duration:.2f} seconds")
    
    # If audio is short, process normally
    if duration <= chunk_length:
        input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
        input_features = input_features.to(device)
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription[0]
    
    # For longer audio, process in chunks
    print(f"Processing audio in {chunk_length}-second chunks...")
    chunk_samples = chunk_length * sr
    full_transcription = []
    
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        if len(chunk) < sr:  # Skip very short chunks
            break
            
        print(f"Processing chunk {i//chunk_samples + 1}/{(len(audio) + chunk_samples - 1)//chunk_samples}")
        
        # Process chunk
        input_features = processor(chunk, sampling_rate=sr, return_tensors="pt").input_features
        input_features = input_features.to(device)
        
        # Generate transcription for this chunk
        with torch.no_grad():  # Save memory
            predicted_ids = model.generate(input_features, max_length=448)
        
        # Decode chunk
        chunk_transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        full_transcription.append(chunk_transcription)
    
    # Combine all chunks
    return " ".join(full_transcription)

# GPU optimization (if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Example usage
if __name__ == "__main__":
    audio_file = "complete_call_whisper.wav"
    try:
        print(f"Transcribing audio file: {audio_file}")
        print("This may take a moment...")
        result = transcribe_audio(audio_file)
        print(f"Transcription: {result}")
    except FileNotFoundError:
        print(f"Error: Audio file '{audio_file}' not found.")
        print("Please make sure the file exists in the current directory.")
    except Exception as e:
        print(f"Error during transcription: {str(e)}")

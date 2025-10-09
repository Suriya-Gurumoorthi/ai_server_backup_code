import nemo.collections.asr as nemo_asr
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    device = 'cuda'
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("CUDA not available, using CPU")

# Load the model and move it to GPU
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
asr_model = asr_model.to(device)

# Run transcription on GPU
print("Starting transcription...")
output = asr_model.transcribe(['Chorus.wav'])
print("Transcription completed!")
print(f"Result: {output[0].text}")
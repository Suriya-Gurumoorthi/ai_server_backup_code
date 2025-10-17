import transformers
import torch
import librosa
import os
import time

# Clear GPU cache first
torch.cuda.empty_cache()

# Simple approach - use CPU without quantization to avoid compatibility issues
pipe = transformers.pipeline(
    model='fixie-ai/ultraVAD', 
    trust_remote_code=True, 
    device="cpu",
    torch_dtype=torch.float32  # Use float32 for better compatibility
)

sr = 16000
wav_path = os.path.join(os.path.dirname(__file__), "ultravox_reply.wav")
audio, sr = librosa.load(wav_path, sr=sr)

turns = [
  {"role": "assistant", "content": "Hi, how are you?"},
]

# Build model inputs via pipeline preprocess
inputs = {"audio": audio, "turns": turns, "sampling_rate": sr}
model_inputs = pipe.preprocess(inputs)

# Move tensors to model device
device = next(pipe.model.parameters()).device
model_inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in model_inputs.items()}

# Forward pass (no generation)
with torch.inference_mode():
  output = pipe.model.forward(**model_inputs, return_dict=True)

# Compute last-audio token position
logits = output.logits  # (1, seq_len, vocab)
audio_pos = int(
  model_inputs["audio_token_start_idx"].item() +
  model_inputs["audio_token_len"].item() - 1
)

# Resolve <|eot_id|> token id and compute probability at last-audio index
token_id = pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
if token_id is None or token_id == pipe.tokenizer.unk_token_id:
  raise RuntimeError("<|eot_id|> not found in tokenizer.")

# Start timing for probability calculation
start_time = time.time()

audio_logits = logits[0, audio_pos, :]
audio_probs = torch.softmax(audio_logits.float(), dim=-1)
eot_prob_audio = audio_probs[token_id].item()

# End timing for probability calculation
end_time = time.time()
probability_time = end_time - start_time

print(f"P(<|eot_id|>) = {eot_prob_audio:.6f}")
print(f"Time taken for probability calculation: {probability_time:.6f} seconds")
threshold = 0.1
if eot_prob_audio > threshold:
  print("Is End of Turn")
else:
  print("Is Not End of Turn")

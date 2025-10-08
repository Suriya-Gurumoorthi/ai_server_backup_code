#!/usr/bin/env python3
"""
Model Loader for Ultravox - Loads the model once and provides it for reuse
"""

import transformers
import torch
import os
import time
import warnings
import gc
warnings.filterwarnings("ignore")

# Global variable to store the loaded model
_ultravox_pipeline = None

def load_ultravox_model():
    """
    Load the Ultravox model with optimized GPU acceleration and 4-bit quantization.
    Returns the pipeline object.
    """
    global _ultravox_pipeline
    
    if _ultravox_pipeline is not None:
        print("✅ Model already loaded, returning existing instance")
        return _ultravox_pipeline
    
    print("🚀 Loading Ultravox Model with Optimized GPU Acceleration...")
    print("=" * 60)
    
    # Set CUDA memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Check GPU availability and optimize settings
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🎮 Using GPU: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
        
        # Clear GPU cache and garbage collect
        torch.cuda.empty_cache()
        gc.collect()
        print(f"🧹 GPU cache and memory cleared")
        
        # Set optimal CUDA settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("⚡ CUDA optimizations enabled")
        
    else:
        device = "cpu"
        print("🖥️  GPU not available, using CPU")
    
    try:
        print(f"\n📦 Loading model with optimized 4-bit quantization on {device.upper()}...")
        start_time = time.time()
        
        # Calculate optimal memory allocation for RTX 3060
        if device == "cuda":
            # Reserve some memory for system and other operations
            available_memory = min(10.5, gpu_memory - 1.0)  # Leave 1GB buffer
            max_memory = f"{available_memory:.1f}GB"
        else:
            max_memory = "32GB"
        
        # Load the model with optimized settings for RTX 3060
        _ultravox_pipeline = transformers.pipeline(
            model='fixie-ai/ultravox-v0_5-llama-3_1-8b', 
            trust_remote_code=True,
            device=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            load_in_4bit=True,  # Use 4-bit quantization for better memory efficiency
            bnb_4bit_compute_dtype=torch.float16 if device == "cuda" else torch.float32,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",  # Use nf4 for 4-bit quantization
            max_memory={0: max_memory} if device == "cuda" else {0: max_memory},
            offload_folder="offload",
            offload_state_dict=True,
            attn_implementation="flash_attention_2" if device == "cuda" else "eager"
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds!")
        
        if device == "cuda":
            gpu_mem_used = torch.cuda.memory_allocated(0) / (1024**3)
            gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            print(f"🎮 Model loaded on GPU - Fast inference enabled!")
            print(f"💾 GPU Memory used: {gpu_mem_used:.1f}GB")
            print(f"💾 GPU Memory reserved: {gpu_mem_reserved:.1f}GB")
            
            # Warm up the model for faster first inference
            print("🔥 Warming up model for optimal performance...")
            warmup_start = time.time()
            
            # Prepare warmup input in correct format
            import numpy as np
            warmup_audio = np.zeros(16000, dtype=np.float32)
            warmup_turns = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "<|audio|>Hello"}
            ]
            
            _ultravox_pipeline(
                {'audio': warmup_audio, 'turns': warmup_turns, 'sampling_rate': 16000},
                max_new_tokens=1, 
                do_sample=False
            )
            warmup_time = time.time() - warmup_start
            print(f"🔥 Warmup completed in {warmup_time:.2f}s")
        else:
            print("💾 Model loaded in RAM - CPU inference")
        
        return _ultravox_pipeline
        
    except Exception as e:
        print(f"❌ Error loading model with GPU: {e}")
        import traceback
        traceback.print_exc()
        
        # Try with even more aggressive memory optimization
        print("\n🔄 Trying with more aggressive memory optimization...")
        try:
            # Clear everything
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Try with 8-bit quantization as fallback
            _ultravox_pipeline = transformers.pipeline(
                model='fixie-ai/ultravox-v0_5-llama-3_1-8b', 
                trust_remote_code=True,
                device=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                load_in_8bit=True,  # Fallback to 8-bit quantization
                max_memory={0: "9GB"} if device == "cuda" else {0: "32GB"},
                offload_folder="offload",
                offload_state_dict=True
            )
            
            print("✅ Model loaded with 8-bit quantization!")
            return _ultravox_pipeline
            
        except Exception as e2:
            print(f"❌ 8-bit quantization also failed: {e2}")
            
            # Final fallback to CPU
            print("\n🔄 Falling back to CPU mode...")
            try:
                # Disable CUDA for fallback
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                torch.cuda.is_available = lambda: False
                
                _ultravox_pipeline = transformers.pipeline(
                    model='fixie-ai/ultravox-v0_5-llama-3_1-8b', 
                    trust_remote_code=True,
                    device="cpu",
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float32,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    max_memory={0: "32GB"},
                    offload_folder="offload",
                    offload_state_dict=True
                )
                
                print("✅ Model loaded on CPU (fallback mode)")
                return _ultravox_pipeline
                
            except Exception as e3:
                print(f"❌ CPU fallback also failed: {e3}")
                return None

def get_ultravox_pipeline():
    """
    Get the loaded Ultravox pipeline. Loads it if not already loaded.
    """
    if _ultravox_pipeline is None:
        return load_ultravox_model()
    return _ultravox_pipeline

def is_model_loaded():
    """
    Check if the model is already loaded.
    """
    return _ultravox_pipeline is not None

def unload_model():
    """
    Unload the model to free memory.
    """
    global _ultravox_pipeline
    if _ultravox_pipeline is not None:
        del _ultravox_pipeline
        _ultravox_pipeline = None
        print("🗑️  Model unloaded from memory")
        return True
    return False

# Load the model when this module is imported
if __name__ == "__main__":
    # Test loading the model
    pipeline = load_ultravox_model()
    if pipeline:
        print("✅ Model loader test successful!")
    else:
        print("❌ Model loader test failed!")

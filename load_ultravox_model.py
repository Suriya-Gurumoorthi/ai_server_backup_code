# pip install transformers peft librosa fastapi uvicorn python-multipart

import transformers
import numpy as np
import librosa
import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Ultravox AI Model API", version="1.0.0")

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model (keeping the original structure)
print("Loading Ultravox model...")
pipe = transformers.pipeline(model='fixie-ai/ultravox-v0_5-llama-3_2-1b', trust_remote_code=True)
print("Model loaded successfully!")

# Pydantic models for request/response
class ConversationTurn(BaseModel):
    role: str
    content: str

class AudioRequest(BaseModel):
    audio_base64: str
    turns: List[ConversationTurn]
    max_new_tokens: int = 30

class AudioResponse(BaseModel):
    response: str
    status: str
    message: str

# Global conversation storage (in production, use a proper database)
conversations: Dict[str, List[Dict[str, str]]] = {}

@app.get("/")
async def root():
    return {"message": "Ultravox AI Model API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/process_audio", response_model=AudioResponse)
async def process_audio(request: AudioRequest):
    """
    Process audio input and return AI response
    """
    try:
        # Decode base64 audio
        audio_data = base64.b64decode(request.audio_base64)
        
        # Load audio using librosa
        audio, sr = librosa.load(io.BytesIO(audio_data), sr=16000)
        
        # Convert turns to the format expected by the model
        turns = [{"role": turn.role, "content": turn.content} for turn in request.turns]
        
        # Process with the model
        result = pipe({
            'audio': audio, 
            'turns': turns, 
            'sampling_rate': sr
        }, max_new_tokens=request.max_new_tokens)
        
        return AudioResponse(
            response=result,
            status="success",
            message="Audio processed successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/process_audio_file", response_model=AudioResponse)
async def process_audio_file(
    audio_file: UploadFile = File(...),
    conversation_id: str = "default",
    max_new_tokens: int = 30
):
    """
    Process uploaded audio file and return AI response
    """
    try:
        # Read audio file
        audio_data = await audio_file.read()
        
        # Load audio using librosa
        audio, sr = librosa.load(io.BytesIO(audio_data), sr=16000)
        
        # Get or create conversation history
        if conversation_id not in conversations:
            conversations[conversation_id] = [
                {
                    "role": "system",
                    "content": "You are a friendly and helpful character. You love to answer questions for people."
                }
            ]
        
        # For audio processing, we should only use system messages and user audio
        # The model expects: system message + user audio (no previous assistant responses)
        system_turns = [turn for turn in conversations[conversation_id] if turn["role"] == "system"]
        
        # Process with the model
        result = pipe({
            'audio': audio, 
            'turns': system_turns, 
            'sampling_rate': sr
        }, max_new_tokens=max_new_tokens)
        
        # Add the response to conversation history (only if it's a valid response)
        if result and isinstance(result, str) and result.strip():
            conversations[conversation_id].append({
                "role": "assistant",
                "content": result
            })
        
        return AudioResponse(
            response=result,
            status="success",
            message="Audio file processed successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio file: {str(e)}")

@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    Get conversation history for a specific conversation ID
    """
    if conversation_id not in conversations:
        return {"conversation_id": conversation_id, "turns": []}
    
    return {"conversation_id": conversation_id, "turns": conversations[conversation_id]}

@app.delete("/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """
    Clear conversation history for a specific conversation ID
    """
    if conversation_id in conversations:
        del conversations[conversation_id]
        return {"message": f"Conversation {conversation_id} cleared successfully"}
    else:
        return {"message": f"Conversation {conversation_id} not found"}

if __name__ == "__main__":
    print("Starting Ultravox AI Model API server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

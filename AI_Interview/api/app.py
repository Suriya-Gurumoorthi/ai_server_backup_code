#!/usr/bin/env python3
"""
FastAPI application for AI Interview Evaluation System.
Provides REST API endpoints for audio file evaluation.
"""

import os
import sys
import uuid
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.processors.audio_processor import process_audio_file
from src.models.ultravox_model import is_model_loaded
from configs.config import AUDIO_CONFIG, PATHS

# Create FastAPI app
app = FastAPI(
    title="AI Interview Evaluation API",
    description="API for evaluating interview audio files using Ultravox model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static files and templates
app.mount("/static", StaticFiles(directory="api/static"), name="static")
templates = Jinja2Templates(directory="api/templates")

# Create upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Store processing status
processing_status = {}

def validate_audio_file(file: UploadFile) -> bool:
    """Validate uploaded audio file"""
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in AUDIO_CONFIG["supported_formats"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Supported formats: {AUDIO_CONFIG['supported_formats']}"
        )
    
    # Check file size (max 50MB)
    if file.size and file.size > 50 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 50MB."
        )
    
    return True

def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file and return file path"""
    # Generate unique filename
    file_ext = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = UPLOAD_DIR / unique_filename
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return str(file_path)

def process_audio_background(file_path: str, job_id: str):
    """Background task to process audio file"""
    try:
        processing_status[job_id] = {"status": "processing", "progress": "Loading audio file..."}
        
        # Process the audio file
        result = process_audio_file(file_path)
        
        if result:
            processing_status[job_id] = {
                "status": "completed",
                "result": result,
                "completed_at": datetime.now().isoformat()
            }
        else:
            processing_status[job_id] = {
                "status": "failed",
                "error": "Failed to process audio file"
            }
            
    except Exception as e:
        processing_status[job_id] = {
            "status": "failed",
            "error": str(e)
        }
    finally:
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main web interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": is_model_loaded(),
        "timestamp": datetime.now().isoformat()
    }



@app.post("/upload")
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and process audio file"""
    try:
        # Validate file
        validate_audio_file(file)
        
        # Save file
        file_path = save_uploaded_file(file)
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Initialize processing status
        processing_status[job_id] = {"status": "queued"}
        
        # Start background processing
        background_tasks.add_task(process_audio_background, file_path, job_id)
        
        return {
            "job_id": job_id,
            "message": "File uploaded successfully. Processing started.",
            "status": "queued"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/status/{job_id}")
async def get_processing_status(job_id: str):
    """Get processing status for a job"""
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_status[job_id]

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Get evaluation results for a completed job"""
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status = processing_status[job_id]
    
    if status["status"] == "completed":
        return {
            "job_id": job_id,
            "status": "completed",
            "result": status["result"],
            "completed_at": status.get("completed_at")
        }
    elif status["status"] == "failed":
        return {
            "job_id": job_id,
            "status": "failed",
            "error": status.get("error")
        }
    else:
        return {
            "job_id": job_id,
            "status": status["status"],
            "message": "Processing in progress"
        }

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its status"""
    if job_id in processing_status:
        del processing_status[job_id]
        return {"message": "Job deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Job not found")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 
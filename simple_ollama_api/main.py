"""
Simple Ollama API - FastAPI application
Provides a /chat endpoint for remote access to local Ollama instance
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import time
from datetime import datetime
from typing import Dict, Any

from config import HOST, PORT, API_KEY, LOG_LEVEL, LOG_FORMAT
from ollama_client import OllamaClient

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Simple Ollama API",
    description="Simple API for remote access to local Ollama instance",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Ollama client
ollama_client = OllamaClient()

# Request/Response models
class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str

class ErrorResponse(BaseModel):
    error: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request):
    """
    Send a prompt to Ollama and return the response
    
    Args:
        request: ChatRequest containing the prompt
        http_request: FastAPI request object for logging
        
    Returns:
        ChatResponse with the model's response
        
    Raises:
        HTTPException: For various error conditions
    """
    start_time = time.time()
    
    # Get client IP for logging
    client_ip = http_request.client.host if http_request.client else "unknown"
    
    # Log the incoming request
    logger.info(f"POST /chat from {client_ip} - prompt: '{request.prompt}'")
    
    try:
        # Send prompt to Ollama
        result = ollama_client.send_prompt(request.prompt)
        
        # Calculate response time
        response_time = int((time.time() - start_time) * 1000)
        
        # Log successful response
        logger.info(f"Response time: {response_time}ms - status: 200")
        
        return ChatResponse(response=result["response"])
        
    except Exception as e:
        # Calculate response time for error
        response_time = int((time.time() - start_time) * 1000)
        
        # Log error
        logger.error(f"Error processing request: {e} - response time: {response_time}ms")
        
        # Return appropriate error response
        if "not available" in str(e):
            raise HTTPException(
                status_code=503,
                detail="Ollama service not available"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Internal server error"
            )

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint
    """
    try:
        is_available = ollama_client.is_available()
        return {
            "status": "healthy" if is_available else "unhealthy",
            "ollama_available": is_available,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "ollama_available": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """
    Middleware to check API key for all requests
    """
    # Skip API key check for health endpoint
    if request.url.path == "/health":
        return await call_next(request)
    
    # Check for API key in headers
    api_key = request.headers.get("x-api-key")
    if api_key != API_KEY:
        logger.warning(f"Unauthorized access attempt from {request.client.host if request.client else 'unknown'}")
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=401,
            content={"error": "Unauthorized"}
        )
    
    return await call_next(request)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Custom exception handler for consistent error responses
    """
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Simple Ollama API on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)

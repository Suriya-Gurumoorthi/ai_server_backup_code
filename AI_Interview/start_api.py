#!/usr/bin/env python3
"""
Startup script for AI Interview Evaluation API.
This script starts the FastAPI server with proper configuration.
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Start the API server"""
    print("="*60)
    print("🚀 STARTING AI INTERVIEW EVALUATION API")
    print("="*60)
    
    # Check if we're in the right directory
    if not Path("api/app.py").exists():
        print("❌ Error: api/app.py not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print("✅ Directories created/verified")
    print("✅ API server starting...")
    print("="*60)
    print("📱 Web Interface: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔧 Health Check: http://localhost:8000/health")
    print("="*60)
    print("Press Ctrl+C to stop the server")
    print("="*60)
    
    # Start the server
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main() 
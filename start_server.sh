#!/bin/bash

# Ultravox API Server Startup Script
# This script installs dependencies and starts the API server

echo "🚀 Starting Ultravox API Server Setup..."
echo "========================================"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python version: $python_version"

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies. Please check the requirements.txt file."
    exit 1
fi

echo "✅ Dependencies installed successfully!"

# Check if the model file exists
if [ ! -f "ultravox_api_server.py" ]; then
    echo "❌ ultravox_api_server.py not found in current directory."
    exit 1
fi

echo "🌐 Starting API server on http://0.0.0.0:8000"
echo "📖 API documentation will be available at http://localhost:8000/docs"
echo "🔍 Health check available at http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================"

# Start the server
python3 ultravox_api_server.py





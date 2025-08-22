#!/bin/bash

# Simple Ollama API Startup Script
echo "🚀 Starting Simple Ollama API..."

# Navigate to the API directory
cd /home/novel/simple_ollama_api

# Activate virtual environment
source venv/bin/activate

# Start the API
echo "📡 API will be available at: http://10.80.2.40:8080"
echo "🔑 API Key: simple_ollama_key_2024"
echo "⏹️  Press Ctrl+C to stop the API"
echo ""

python main.py

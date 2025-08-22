"""
Configuration for Simple Ollama API
"""

# Server Configuration
HOST = "10.80.2.40"  # Your server's IP address
PORT = 8080

# Ollama Configuration
OLLAMA_HOST = "localhost"
OLLAMA_PORT = 11434
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
MODEL_NAME = "llama3.2:3b"

# Security
API_KEY = "simple_ollama_key_2024"  # Change this to your preferred API key

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"

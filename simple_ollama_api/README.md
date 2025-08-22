# Simple Ollama API

A lightweight FastAPI application that exposes your local Ollama LLaMA 3B model via a simple `/chat` endpoint for remote access.

## 🎯 Features

- ✅ **Single endpoint**: `/chat` for all interactions
- ✅ **Simple authentication**: API key in header
- ✅ **CORS enabled**: Works from any origin
- ✅ **Error handling**: Proper HTTP status codes
- ✅ **Request logging**: Timestamp, IP, prompt, response time
- ✅ **Health check**: `/health` endpoint
- ✅ **No complex setup**: Just run and use

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API
```bash
python main.py
```

The API will start on `http://10.80.2.40:8080`

### 3. Test Locally
```bash
curl -X POST http://10.80.2.40:8080/chat \
  -H "Content-Type: application/json" \
  -H "x-api-key: simple_ollama_key_2024" \
  -d '{"prompt": "Hello, how are you?"}'
```

## 📋 API Reference

### POST /chat

Send a prompt to the LLaMA 3B model.

**Request:**
```json
{
  "prompt": "What is artificial intelligence?"
}
```

**Response:**
```json
{
  "response": "Artificial intelligence (AI) refers to..."
}
```

**Headers Required:**
- `Content-Type: application/json`
- `x-api-key: simple_ollama_key_2024`

### GET /health

Check if the API and Ollama are available.

**Response:**
```json
{
  "status": "healthy",
  "ollama_available": true,
  "timestamp": "2025-08-11T07:15:00Z"
}
```

## 🔐 Security

- **API Key**: Required for all requests (except `/health`)
- **Default Key**: `simple_ollama_key_2024` (change in `config.py`)
- **CORS**: Enabled for all origins (can be restricted later)

## 🌐 Cross-Server Usage

### From Server B (Remote Server)

```bash
# Set variables
export SERVER_A_IP="192.168.1.100"  # Replace with Server A's IP
export API_KEY="simple_ollama_key_2024"

# Test the API
curl -X POST "http://$SERVER_A_IP:8080/chat" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{"prompt": "Hello from Server B!"}'
```

### Using Python

```python
import requests

url = "http://SERVER_A_IP:8080/chat"
headers = {
    "Content-Type": "application/json",
    "x-api-key": "simple_ollama_key_2024"
}
data = {"prompt": "What is machine learning?"}

response = requests.post(url, headers=headers, json=data)
print(response.json()["response"])
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Server settings
HOST = "0.0.0.0"  # Bind to all interfaces
PORT = 8080       # API port

# Ollama settings
OLLAMA_HOST = "localhost"
OLLAMA_PORT = 11434
MODEL_NAME = "llama3.2:3b"

# Security
API_KEY = "simple_ollama_key_2024"  # Change this!

# Logging
LOG_LEVEL = "INFO"
```

## 📊 Logging

The API logs all requests with:
- Timestamp
- Client IP
- Prompt content
- Response time
- Status codes

Example log output:
```
[2025-08-11 07:15:00] INFO: POST /chat from 192.168.1.100 - prompt: 'Hello, how are you?'
[2025-08-11 07:15:03] INFO: Response time: 2500ms - status: 200
```

## 🚨 Error Handling

| Status Code | Description | Example |
|-------------|-------------|---------|
| 200 | Success | Normal response |
| 400 | Bad Request | Missing "prompt" field |
| 401 | Unauthorized | Invalid/missing API key |
| 503 | Service Unavailable | Ollama not running |
| 500 | Internal Server Error | Unexpected errors |

## 🛠️ Troubleshooting

### API Won't Start
```bash
# Check if port is available
netstat -tlnp | grep 8080

# Check Python dependencies
pip list | grep fastapi
```

### Ollama Connection Failed
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Check health endpoint
curl http://localhost:8080/health
```

### API Key Issues
```bash
# Test without API key (should fail)
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test"}'

# Test with wrong API key (should fail)
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -H "x-api-key: wrong_key" \
  -d '{"prompt": "test"}'
```

## 🔄 Deployment

### Development
```bash
python main.py
```

### Production
```bash
# Run in background
nohup python main.py > api.log 2>&1 &

# Or use systemd service
sudo systemctl enable simple-ollama-api
sudo systemctl start simple-ollama-api
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["python", "main.py"]
```

## 📁 Project Structure

```
simple_ollama_api/
├── main.py              # FastAPI application
├── ollama_client.py     # Ollama communication
├── config.py           # Configuration
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## 🎉 Success!

Your Simple Ollama API is now ready to serve LLaMA 3B responses to remote servers!

**Flow**: Server B → API → Ollama → API → Server B

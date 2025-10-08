# Ultravox API Server for Vicidial Integration

This project converts the Ultravox AI model into an HTTP API server that can be integrated with Vicidial or any other call center system for human-to-AI conversations.

## Features

- 🎤 **Audio Processing**: Accepts base64-encoded audio and processes it with the Ultravox model
- 💬 **Text Conversations**: Handles text-only conversations
- 🔄 **Conversation History**: Maintains context across multiple turns
- 🌐 **HTTP API**: RESTful API endpoints for easy integration
- 📊 **Health Monitoring**: Built-in health checks and memory monitoring
- 🔧 **Configurable**: Adjustable parameters like temperature and max tokens

## Quick Start

### 1. Install Dependencies

```bash
# Make the startup script executable
chmod +x start_server.sh

# Run the startup script (installs dependencies and starts server)
./start_server.sh
```

Or manually:

```bash
pip3 install -r requirements.txt
python3 ultravox_api_server.py
```

### 2. Access the API

- **API Server**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## API Endpoints

### Health Check
```http
GET /health
```

### Text Conversation
```http
POST /chat/text
Content-Type: application/json

{
  "turns": [
    {
      "role": "system",
      "content": "You are a helpful customer service assistant."
    },
    {
      "role": "user", 
      "content": "Hello, I need help with my order."
    }
  ],
  "max_new_tokens": 100,
  "temperature": 0.7
}
```

### Audio Conversation
```http
POST /chat/audio
Content-Type: application/json

{
  "audio_base64": "base64_encoded_audio_data",
  "turns": [
    {
      "role": "system",
      "content": "You are a helpful assistant. Transcribe and respond to the audio."
    }
  ],
  "sampling_rate": 16000,
  "max_new_tokens": 100,
  "temperature": 0.7
}
```

### Generic Conversation
```http
POST /chat/conversation
Content-Type: application/json

{
  "turns": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello!"
    }
  ],
  "max_new_tokens": 100,
  "temperature": 0.7
}
```

## Vicidial Integration

### Example Integration Code

```python
import requests
import base64

class VicidialAIIntegration:
    def __init__(self, api_url="http://your-server:8000"):
        self.api_url = api_url
    
    def process_customer_audio(self, audio_data, conversation_history):
        """Process customer audio and get AI response"""
        
        # Encode audio to base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Prepare request
        payload = {
            "audio_base64": audio_base64,
            "turns": conversation_history,
            "sampling_rate": 16000,
            "max_new_tokens": 50,
            "temperature": 0.7
        }
        
        # Send request
        response = requests.post(
            f"{self.api_url}/chat/audio",
            json=payload
        )
        
        return response.json()
    
    def process_text_message(self, message, conversation_history):
        """Process text message and get AI response"""
        
        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": message
        })
        
        # Prepare request
        payload = {
            "turns": conversation_history,
            "max_new_tokens": 50,
            "temperature": 0.7
        }
        
        # Send request
        response = requests.post(
            f"{self.api_url}/chat/text",
            json=payload
        )
        
        result = response.json()
        
        if result.get("success"):
            # Add AI response to history
            conversation_history.append({
                "role": "assistant",
                "content": result["data"]["response"]
            })
        
        return result
```

### Integration Steps

1. **Deploy the API server** on a machine accessible to your Vicidial server
2. **Configure network access** between Vicidial and the API server
3. **Implement the integration code** in your Vicidial custom scripts
4. **Test the integration** using the provided test client

## Testing

Run the test client to verify the API is working:

```bash
python3 test_api_client.py
```

This will:
- Test health checks
- Test text conversations
- Test audio processing
- Simulate a customer service call

## Configuration

### Environment Variables

You can set these environment variables to customize the server:

```bash
export ULTRAVOX_HOST="0.0.0.0"  # Server host
export ULTRAVOX_PORT="8000"     # Server port
export ULTRAVOX_MODEL="fixie-ai/ultravox-v0_5-llama-3_1-8b"  # Model name
```

### Model Parameters

- **max_new_tokens**: Maximum number of tokens to generate (default: 100)
- **temperature**: Controls randomness in responses (0.0-1.0, default: 0.7)
- **sampling_rate**: Audio sampling rate in Hz (default: 16000)

## Production Deployment

### Using systemd (Linux)

Create a service file `/etc/systemd/system/ultravox-api.service`:

```ini
[Unit]
Description=Ultravox API Server
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/ultravox-api
ExecStart=/usr/bin/python3 ultravox_api_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl enable ultravox-api
sudo systemctl start ultravox-api
```

### Using Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "ultravox_api_server.py"]
```

Build and run:

```bash
docker build -t ultravox-api .
docker run -p 8000:8000 ultravox-api
```

## Troubleshooting

### Common Issues

1. **Model loading fails**: Check internet connection and disk space
2. **Memory issues**: The model requires significant RAM (8GB+ recommended)
3. **Audio processing errors**: Ensure audio is properly encoded in base64
4. **Network connectivity**: Verify firewall settings and network access

### Logs

The server logs important information including:
- Model loading status
- Request processing
- Error messages
- Memory usage

### Performance Monitoring

Monitor the `/memory` endpoint to track memory usage:

```bash
curl http://localhost:8000/memory
```

## Security Considerations

1. **Network Security**: Use HTTPS in production
2. **Authentication**: Implement API key authentication
3. **Rate Limiting**: Add rate limiting for production use
4. **Input Validation**: Validate all inputs before processing
5. **CORS Configuration**: Configure CORS properly for your domain

## Support

For issues and questions:
1. Check the logs for error messages
2. Verify the model is loaded correctly
3. Test with the provided test client
4. Check network connectivity between servers

## License

This project is provided as-is for integration purposes.

# Ultravox WebSocket Server

A modular WebSocket server for real-time audio processing using Ultravox and Piper TTS.

## Architecture

The server is organized into the following modules:

- **`config.py`** - Configuration constants and settings
- **`utils.py`** - Utility functions and logging setup
- **`models.py`** - AI model management (Ultravox and Piper TTS)
- **`audio_processor.py`** - Audio processing and TTS generation
- **`websocket_handler.py`** - WebSocket connection and message handling
- **`server.py`** - Main server orchestration and entry point

## Features

- Real-time audio transcription using Ultravox
- Text-to-speech generation using Piper TTS
- Chunked audio streaming for large responses
- GPU acceleration support
- Modular, maintainable codebase
- Comprehensive logging and error handling

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download required model files:
   - Ultravox model will be downloaded automatically
   - Piper TTS model files need to be placed in the server directory

## Usage

Run the server:
```bash
python server.py
```

The server will start on `0.0.0.0:8000` by default.

## API

### WebSocket Messages

#### Text/JSON Messages:
- `{"type": "transcribe"}` - Request audio transcription
- `{"type": "features"}` - Request audio analysis with speaker features
- `{"type": "tts"}` - Request text-to-speech generation
- `{"type": "voices"}` - Get available TTS voices

#### Binary Messages:
- Raw audio data (WAV format, 16kHz sample rate)

### Response Format

- **Transcription**: Plain text response
- **Features**: Text with speaker analysis
- **TTS**: Chunked audio data with metadata
- **Voices**: JSON list of available voices

## Configuration

Edit `config.py` to modify:
- Server host/port
- Model settings
- Audio processing parameters
- Conversation prompts

## Development

The modular structure makes it easy to:
- Add new AI models
- Implement new audio processing features
- Extend WebSocket functionality
- Modify conversation behavior

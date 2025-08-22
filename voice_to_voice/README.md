# Voice-to-Voice AI System

A comprehensive voice-to-voice AI system that enables real-time conversations with AI agents using Piper TTS and Ultravox STT.

## System Overview

This system provides a complete voice-to-voice AI conversation experience with the following features:

1. **Welcome Message**: AI introduces itself based on the assigned role (e.g., real estate agent)
2. **Real-time Speech Recognition**: Uses Ultravox for live audio processing
3. **Contextual Understanding**: Processes audio embeddings to extract user intent and details
4. **Text-to-Speech Response**: Uses Piper TTS to generate natural voice responses
5. **Interruption Handling**: Supports mid-conversation interruptions
6. **Call Management**: Handles call termination and conversation summaries
7. **Web Interface**: Browser-based testing interface

## Directory Structure

```
voice_to_voice/
├── main.py                          # Main application entry point
├── app.py                           # Web application entry point
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── config/                          # Configuration files
│   ├── settings.py                  # System settings and parameters
│   └── roles.py                     # AI role definitions and prompts
├── src/                             # Source code
│   ├── core/                        # Core system components
│   │   ├── audio_processor.py       # Audio processing utilities
│   │   ├── conversation_manager.py  # Main conversation orchestrator
│   │   ├── voice_session.py         # Session management
│   │   ├── audio_stream_manager.py  # Real-time audio stream handling
│   │   ├── audio_processing/        # Audio processing modules
│   │   ├── embedding/               # Audio embedding processing
│   │   └── context_management/      # Conversation context management
│   ├── tts/                         # Text-to-Speech components
│   │   ├── piper_tts.py             # Piper TTS integration
│   │   ├── voice_generator.py       # Voice synthesis orchestration
│   │   ├── piper_integration/       # Piper-specific modules
│   │   └── voice_synthesis/         # Voice synthesis utilities
│   ├── stt/                         # Speech-to-Text components
│   │   ├── ultravox_stt.py          # Ultravox STT integration
│   │   ├── speech_recognizer.py     # Speech recognition orchestration
│   │   ├── ultravox_integration/    # Ultravox-specific modules
│   │   └── speech_recognition/      # Speech recognition utilities
│   ├── conversation/                # Conversation management
│   │   ├── conversation_flow.py     # Conversation flow control
│   │   ├── role_handler.py          # AI role management
│   │   ├── prompt_manager.py        # Prompt handling and generation
│   │   ├── flow_control/            # Flow control modules
│   │   ├── role_management/         # Role-specific modules
│   │   └── prompt_handling/         # Prompt processing modules
│   ├── interruption/                # Interruption handling
│   │   ├── interruption_detector.py # Interruption detection
│   │   ├── interruption_handler.py  # Interruption response handling
│   │   ├── detection/               # Detection algorithms
│   │   └── handling/                # Handling strategies
│   ├── reporting/                   # Reporting and storage
│   │   ├── report_generator.py      # Conversation report generation
│   │   ├── conversation_storage.py  # Conversation data storage
│   │   ├── generation/              # Report generation modules
│   │   └── storage/                 # Storage utilities
│   ├── web/                         # Web interface
│   │   ├── app.py                   # Web application
│   │   ├── routes.py                # API routes
│   │   ├── websocket_handler.py     # WebSocket handling
│   │   ├── static/                  # Static web assets
│   │   │   ├── css/
│   │   │   │   └── style.css        # Web interface styles
│   │   │   └── js/
│   │   │       ├── voice_interface.js    # Voice interface logic
│   │   │       └── websocket_client.js   # WebSocket client
│   │   ├── templates/              # HTML templates
│   │   │   └── index.html          # Main web interface
│   │   └── api/                    # API modules
│   └── utils/                      # Utility functions
│       ├── audio_utils.py          # Audio processing utilities
│       ├── text_utils.py           # Text processing utilities
│       └── logger.py               # Logging utilities
├── models/                         # Model files and configurations
├── data/                           # Data storage
│   ├── audio/                      # Audio files
│   ├── conversations/              # Conversation logs
│   └── reports/                    # Generated reports
├── tests/                          # Test files
│   ├── test_tts.py                 # TTS testing
│   ├── test_stt.py                 # STT testing
│   └── test_conversation.py        # Conversation testing
└── docs/                           # Documentation
```

## Key Components

### 1. Core System (`src/core/`)
- **Audio Processing**: Real-time audio capture and processing
- **Conversation Management**: Orchestrates the entire conversation flow
- **Session Management**: Handles user sessions and state
- **Context Management**: Maintains conversation context and history

### 2. Text-to-Speech (`src/tts/`)
- **Piper Integration**: Direct integration with Piper TTS engine
- **Voice Synthesis**: Converts AI responses to natural speech
- **Voice Generation**: Manages voice quality and characteristics

### 3. Speech-to-Text (`src/stt/`)
- **Ultravox Integration**: Direct integration with Ultravox STT engine
- **Speech Recognition**: Real-time speech-to-text conversion
- **Audio Embedding**: Processes audio for intent understanding

### 4. Conversation Management (`src/conversation/`)
- **Flow Control**: Manages conversation turn-taking
- **Role Management**: Handles different AI roles (real estate agent, etc.)
- **Prompt Handling**: Generates contextual responses

### 5. Interruption Handling (`src/interruption/`)
- **Detection**: Detects when user interrupts AI
- **Handling**: Manages interruption responses and flow

### 6. Reporting (`src/reporting/`)
- **Report Generation**: Creates conversation summaries
- **Storage**: Stores conversation data and reports

### 7. Web Interface (`src/web/`)
- **Web Application**: Flask/FastAPI web server
- **WebSocket Handling**: Real-time communication
- **Static Assets**: CSS, JavaScript, and HTML templates

## Workflow

1. **Initialization**: System loads with specified AI role
2. **Welcome Message**: AI introduces itself using TTS
3. **Listening Phase**: System listens for user input via STT
4. **Processing**: Audio is converted to text and analyzed
5. **Response Generation**: AI generates contextual response
6. **Voice Output**: Response is converted to speech via TTS
7. **Interruption Handling**: System detects and handles interruptions
8. **Call Termination**: System ends conversation and generates report

## Configuration

- **`config/settings.py`**: System parameters, audio settings, model paths
- **`config/roles.py`**: AI role definitions and behavior patterns

## Data Storage

- **`data/audio/`**: Temporary audio files
- **`data/conversations/`**: Conversation logs and transcripts
- **`data/reports/`**: Generated conversation reports

## Testing

- **`tests/`**: Unit tests for each component
- **Web Interface**: Browser-based testing interface

## Dependencies

- Piper TTS for text-to-speech
- Ultravox for speech-to-text
- Web framework (Flask/FastAPI)
- WebSocket support
- Audio processing libraries

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Configure settings in `config/settings.py`
3. Run the web interface: `python app.py`
4. Access the interface at `http://localhost:5000`
5. Start a voice conversation with the AI

## Development

- Follow the modular structure for easy maintenance
- Add new roles in `config/roles.py`
- Extend functionality by adding new modules
- Test components individually using the test suite

# Vicidial Bridge - Modular Voice Bot System

This is a modular refactoring of the original `ai_voice_bot_ultravox_local.py` monolithic script. The code has been organized into logical modules for better maintainability, debugging, and extensibility.

## Module Structure

### Core Modules

- **`config.py`** - Configuration constants, environment variables, and system prompts
- **`audio_processing.py`** - Audio format conversion and AudioSocket protocol handling
- **`voice_activity_detection.py`** - Voice Activity Detection (VAD) implementation
- **`text_to_speech.py`** - Piper TTS integration and audio format conversion
- **`call_recording.py`** - Complete call recording and metadata generation
- **`websocket_client.py`** - WebSocket client for local model server communication
- **`bridge.py`** - Main bridge class that orchestrates the entire system
- **`bridge_methods.py`** - Helper methods for the bridge class
- **`main.py`** - Main entry point and logging setup

### Entry Points

- **`run_bridge.py`** - Executable launcher script
- **`__init__.py`** - Package initialization and exports

## Features

- **Modular Architecture**: Clean separation of concerns for easier maintenance
- **Voice Activity Detection**: Intelligent filtering of silence and noise
- **Real-time Audio Processing**: Optimized for low-latency streaming
- **Complete Call Recording**: Full conversation capture with metadata
- **High-quality TTS**: Piper neural voices for natural speech synthesis
- **WebSocket Communication**: Efficient communication with local AI models
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## Usage

### Running the Bridge

```bash
# Using the launcher script
python3 /usr/vicidial_bridge/run_bridge.py

# Or directly with the main module
python3 -m vicidial_bridge.main

# Or using the package
python3 -c "from vicidial_bridge import VicialLocalModelBridge; import asyncio; asyncio.run(VicialLocalModelBridge().start_server())"
```

### Configuration

All configuration is managed through `config.py`. Key settings include:

- **Network Configuration**: AudioSocket and WebSocket endpoints
- **Audio Settings**: Sample rates, buffer sizes, processing intervals
- **VAD Parameters**: Energy thresholds, silence detection
- **Recording Options**: Call recording, debug audio, metadata
- **Production Settings**: Log levels, performance optimization

### Directory Structure

```
/usr/vicidial_bridge/
├── __init__.py              # Package initialization
├── config.py                # Configuration and constants
├── audio_processing.py      # Audio processing and AudioSocket protocol
├── voice_activity_detection.py  # VAD implementation
├── text_to_speech.py        # TTS integration
├── call_recording.py        # Call recording functionality
├── websocket_client.py      # WebSocket client
├── bridge.py                # Main bridge class
├── bridge_methods.py        # Bridge helper methods
├── main.py                  # Main entry point
├── run_bridge.py            # Executable launcher
└── README.md                # This file
```

## Compatibility

This modular version maintains **100% functional compatibility** with the original monolithic script. All features, configurations, and behaviors are preserved:

- Same AudioSocket protocol handling
- Identical audio processing pipeline
- Same VAD implementation and parameters
- Identical TTS integration
- Same call recording functionality
- Same WebSocket communication
- Identical logging and monitoring

## Benefits of Modular Structure

1. **Easier Debugging**: Issues can be isolated to specific modules
2. **Better Testing**: Individual components can be unit tested
3. **Improved Maintainability**: Changes are localized to relevant modules
4. **Enhanced Readability**: Code is organized by functionality
5. **Extensibility**: New features can be added as separate modules
6. **Reusability**: Components can be reused in other projects

## Dependencies

The modular version uses the same dependencies as the original:

- `asyncio` - Asynchronous programming
- `websockets` - WebSocket communication
- `numpy` - Audio processing
- `wave` - Audio file handling
- `piper` - Text-to-speech synthesis
- `python-dotenv` - Environment variable management

## Migration from Original

To migrate from the original monolithic script:

1. The modular version can run alongside the original (different ports)
2. Configuration remains the same (uses same `.env` file)
3. All audio files and recordings are saved to the same locations
4. Logging format and locations are identical
5. No changes needed to Vicial or local model server configuration

## Support

For issues or questions about the modular version, refer to the original script's documentation and ensure all dependencies are properly installed.



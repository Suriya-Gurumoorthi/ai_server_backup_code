# Audio Socket Migration - Vicidial Bridge Updates

## Overview
This document outlines the changes made to the Vicidial Bridge to support the new audio socket communication from the Ultravox server.

## Key Changes Made

### 1. WebSocket Client Updates (`websocket_client.py`)
- **Enhanced Audio Response Handling**: Updated `_listen_for_responses()` to prioritize audio responses from Ultravox
- **Direct Audio Processing**: Now handles WAV audio data directly from the server without requiring TTS conversion
- **Fallback Support**: Maintains support for text responses (legacy mode) while prioritizing audio responses

### 2. Bridge Methods Updates (`bridge_methods.py`)
- **Primary Audio Path**: Updated `_process_local_model_responses()` to treat audio responses as the primary communication method
- **Turn Control**: Enhanced barge-in support with proper turn validation for audio responses
- **Efficient Processing**: Streamlined audio processing to handle direct audio from Ultravox

### 3. Configuration Updates (`config.py`)
- **Audio-First Approach**: Added documentation about the new primary audio path
- **TTS Fallback**: Clarified that TTS is now a fallback for text responses only

### 4. Logging Updates
- **Bridge Server** (`bridge.py`): Updated startup messages to reflect audio-first processing
- **Main Entry Point** (`main.py`): Updated logging to show direct audio processing as primary path

## Communication Flow

### Before (Text-Based)
1. Vicidial Bridge → Ultravox Server: Audio data
2. Ultravox Server → Vicidial Bridge: Text response
3. Vicidial Bridge: Convert text to speech using Piper TTS
4. Vicidial Bridge → ViciDial: Audio playback

### After (Audio Socket)
1. Vicidial Bridge → Ultravox Server: Audio data
2. Ultravox Server → Vicidial Bridge: **Direct audio response**
3. Vicidial Bridge: Convert audio format (16kHz → 8kHz)
4. Vicidial Bridge → ViciDial: Audio playback

## Benefits

1. **Reduced Latency**: Eliminates TTS conversion step in the bridge
2. **Better Quality**: Uses Ultravox's native TTS output directly
3. **Simplified Processing**: Fewer conversion steps in the audio pipeline
4. **Maintained Compatibility**: Still supports text responses as fallback

## Technical Details

### Audio Format Handling
- **Input**: 8kHz PCM from ViciDial → 16kHz PCM to Ultravox
- **Output**: 16kHz WAV from Ultravox → 8kHz PCM to ViciDial
- **Conversion**: Uses existing upsampling/downsampling methods

### Turn Management
- **Barge-in Support**: Maintains existing turn-based interruption system
- **Race Condition Prevention**: Prevents multiple audio streams from conflicting
- **Call State Management**: Preserves existing call lifecycle management

### Error Handling
- **Graceful Degradation**: Falls back to text processing if audio processing fails
- **Connection Management**: Maintains existing WebSocket connection handling
- **Logging**: Enhanced logging for audio-first processing

## Migration Notes

- **Backward Compatibility**: All existing functionality is preserved
- **No Breaking Changes**: Existing configuration and deployment remains the same
- **Enhanced Performance**: Better latency and audio quality
- **Maintained Features**: All VAD, barge-in, and recording features work as before

## Testing Recommendations

1. **Audio Quality**: Verify audio quality is maintained or improved
2. **Latency**: Measure end-to-end latency improvements
3. **Barge-in**: Test interruption functionality still works correctly
4. **Fallback**: Ensure text responses still work if audio processing fails
5. **Recording**: Verify call recording captures both sides correctly

## Configuration

No configuration changes are required. The bridge automatically detects and handles both audio and text responses from the Ultravox server.

## Deployment

The updated bridge can be deployed alongside the new Ultravox server without any additional configuration changes.

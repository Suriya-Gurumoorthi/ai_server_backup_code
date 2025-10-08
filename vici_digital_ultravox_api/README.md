# Vicidial-Ultravox Bridge

This bridge connects Vicidial (8kHz audio) with Ultravox AI (48kHz audio) by handling real-time audio sample rate conversion.

## Features

- **Upsampling**: Converts 8kHz Vicidial audio to 48kHz for Ultravox
- **Downsampling**: Converts 48kHz Ultravox audio back to 8kHz for Vicidial
- **Real-time processing**: Handles bidirectional audio streams
- **Anti-aliasing filters**: Prevents audio artifacts during conversion
- **Connection management**: Handles multiple concurrent connections

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your Ultravox API key in `openai.py`:
```python
ULTRAVOX_API_KEY = "your-api-key-here"
```

## Usage

1. Start the bridge server:
```bash
python openai.py
```

2. The server will listen on `0.0.0.0:9092` for Vicidial connections

3. Configure Vicidial to connect to this bridge for AI voice interactions

## Audio Processing Details

### Upsampling (8kHz → 48kHz)
- Inserts 5 zero samples between each original sample
- Applies low-pass filter to interpolate missing values
- Scales amplitude to maintain proper levels

### Downsampling (48kHz → 8kHz)
- Applies anti-aliasing low-pass filter
- Takes every 6th sample to reduce sample rate
- Maintains audio quality while preventing aliasing

## Configuration

- `VICIDIAL_SAMPLE_RATE`: 8000 Hz (standard for telephony)
- `ULTRAVOX_SAMPLE_RATE`: 48000 Hz (Ultravox requirement)
- `BYTES_PER_FRAME`: 320 bytes (160 samples × 2 bytes per sample)
- `UPSAMPLE_FACTOR`: 6 (48000/8000)

## Error Handling

The bridge includes comprehensive error handling for:
- Network connection failures
- Audio processing errors
- Ultravox API errors
- Connection cleanup

## Logging

All operations are logged with appropriate levels:
- INFO: Connection events and normal operations
- ERROR: Failures and exceptions
- DEBUG: Detailed audio processing information (if enabled)

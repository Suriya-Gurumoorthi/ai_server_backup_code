# VICIdial Bridge Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           VICIdial Bridge System Architecture                    │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    AudioSocket     ┌──────────────────┐    WebSocket     ┌─────────────────┐
│   VICIdial      │◄──────────────────►│  VICIdial Bridge │◄────────────────►│  Local AI Model │
│   Server        │    (Port 9092)     │                  │   (Port 8000)    │   Server        │
│   (Asterisk)    │                    │                  │                  │   (Ultravox)    │
└─────────────────┘                    └──────────────────┘                  └─────────────────┘
         │                                       │                                    │
         │                                       │                                    │
         ▼                                       ▼                                    ▼
┌─────────────────┐                    ┌──────────────────┐                  ┌─────────────────┐
│   Caller        │                    │  Audio Processing│                  │  Piper TTS      │
│   (Phone)       │                    │  Pipeline        │                  │  Engine         │
└─────────────────┘                    └──────────────────┘                  └─────────────────┘
                                               │
                                               ▼
                                    ┌──────────────────┐
                                    │  Voice Activity  │
                                    │  Detection (VAD) │
                                    └──────────────────┘
                                               │
                                               ▼
                                    ┌──────────────────┐
                                    │  Call Recording  │
                                    │  System          │
                                    └──────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Data Flow Process                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

1. Call Initiation:
   Caller → VICIdial → Extension 8888 → AudioSocket Connection

2. Audio Processing:
   VICIdial (8kHz) → Bridge → Upsample to 16kHz → Local AI Model

3. AI Response:
   Local AI Model → Text/Audio → Bridge → TTS (if text) → Downsample to 8kHz → VICIdial

4. Call Recording:
   All audio streams → CallRecorder → WAV files + Metadata

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Key Technologies                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

• AudioSocket Protocol: Real-time audio streaming with Asterisk
• WebSocket: Bidirectional communication with AI model server
• NumPy: High-performance audio processing and format conversion
• Piper TTS: Neural voice synthesis for natural speech
• Voice Activity Detection: Advanced filtering to prevent false triggers
• Asyncio: Concurrent processing for low-latency real-time communication
• WAV Processing: Complete call recording and audio analysis

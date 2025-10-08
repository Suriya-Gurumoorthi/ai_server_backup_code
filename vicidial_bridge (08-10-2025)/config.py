"""
Configuration module for Vicidial Bridge

Contains all configuration constants, environment variables, and system prompts.
"""

import os
import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback if python-dotenv is not installed
    def load_dotenv():
        pass
    load_dotenv()

# ==================== NETWORK CONFIGURATION ====================
VICIAL_AUDIOSOCKET_HOST = "0.0.0.0"
VICIAL_AUDIOSOCKET_PORT = 9092  # Changed from 9093 to 9092

# Local Model Server Configuration
LOCAL_MODEL_HOST = "10.80.2.40"
LOCAL_MODEL_PORT = 8000
# WebSocket endpoint for local model server
LOCAL_MODEL_WS_URL = f"ws://{LOCAL_MODEL_HOST}:{LOCAL_MODEL_PORT}"

# ==================== AUDIO CONFIGURATION ====================
VICIAL_SAMPLE_RATE = 8000   # Vicial uses 8kHz
LOCAL_MODEL_SAMPLE_RATE = 16000  # Local model expects 16kHz
UPSAMPLE_FACTOR = LOCAL_MODEL_SAMPLE_RATE // VICIAL_SAMPLE_RATE  # 2x
DOWNSAMPLE_FACTOR = UPSAMPLE_FACTOR
BYTES_PER_FRAME = 320  # 160 samples * 2 bytes per sample (16-bit audio)
AUDIO_CHANNELS = 1  # Mono audio

# Audio streaming timing configuration
AUDIO_CHUNK_SIZE = 160  # Bytes per chunk (10ms at 8kHz) - Reduced for faster playback
AUDIO_SILENCE_PADDING_MS = 40  # Silence padding before each TTS response (ms)
AUDIO_TIMING_TOLERANCE_MS = 100  # Maximum timing lag before warning (ms)

# Audio playback speed control
AUDIO_SPEED_MULTIPLIER = 1.4  # Speed multiplier (1.0 = normal, >1.0 = faster, <1.0 = slower)

# ==================== CALL ROUTING CONFIGURATION ====================
TARGET_EXTENSION = "8888"  # Changed from 9999 to 8888

# ==================== AUDIO LOGGING CONFIGURATION ====================
SAVE_AI_AUDIO = True  # Save AI response audio files
AI_AUDIO_DIR = "/usr/ai_responses"  # Directory to save AI audio responses

# DEBUG AUDIO RECORDING: Save all incoming audio for analysis
SAVE_DEBUG_AUDIO = True  # Save all incoming audio for debugging
DEBUG_AUDIO_DIR = "/usr/debug_audio"  # Directory to save debug audio files

# COMPLETE CALL RECORDING: Save entire call conversations
SAVE_COMPLETE_CALLS = True  # Save complete call recordings
CALL_RECORDINGS_DIR = "/usr/call_recordings"  # Directory to save complete call recordings
SAVE_SEPARATE_TRACKS = True  # Save separate caller and AI audio tracks
SAVE_CALL_METADATA = True  # Save detailed call metadata files

# ==================== LOCAL MODEL CONFIGURATION ====================
# Note: Model configuration will be handled by the local server
# LOCAL_MODEL_VOICE removed - voice output configured via local server

# RAG Configuration - Local model may have different RAG setup
LOCAL_MODEL_CORPUS_ID = "local_corpus"  # Local model corpus identifier

# RAG Parameters - Adjust these based on your local model needs
RAG_MAX_RESULTS = 5        # Number of document chunks to retrieve
RAG_MIN_SCORE = 0.75      # Minimum relevance score (0.0 to 1.0)

# ==================== LATENCY OPTIMIZATION ====================
# Reduced audio processing intervals for near real-time streaming
AUDIO_PROCESS_INTERVAL = 0.1    # 100ms (increased for interview stability)
AUDIO_BUFFER_SIZE_MS = 100      # 100ms buffer size for interview stability
MIN_AUDIO_CHUNK_SIZE = 1600     # Minimum bytes before processing (50ms at 16kHz)
MAX_AUDIO_CHUNK_SIZE = 3200     # Maximum bytes to accumulate (100ms at 16kHz)

# ==================== VOICE ACTIVITY DETECTION (VAD) CONFIGURATION ====================
VAD_ENABLED = True              # Enable Voice Activity Detection
VAD_ENERGY_THRESHOLD = 500      # Energy threshold for speech detection (0-32767)
VAD_SILENCE_DURATION_MS = 1500  # Silence duration to end speech (milliseconds) - Increased for interview thinking pauses
VAD_MIN_SPEECH_DURATION_MS = 500 # Minimum speech duration to process (milliseconds) - Increased to ignore brief sounds
VAD_DEBUG_LOGGING = True        # Enable detailed VAD logging

# Enhanced VAD parameters to prevent mic taps and noise from triggering speech detection
VAD_HIGH_PASS_CUTOFF = 250.0    # High-pass filter cutoff frequency (Hz) to suppress low-frequency taps
VAD_MIN_CONSECUTIVE_FRAMES = 8  # Minimum consecutive frames required for speech detection (debounce) - Increased for stability
VAD_SPECTRAL_FLATNESS_THRESHOLD = 1.0  # Spectral flatness threshold (higher = more noise-like)

# Real-time barge-in configuration for immediate TTS interruption
VAD_BARGE_IN_CONSECUTIVE_FRAMES = 2  # Lower threshold for real-time barge-in detection (more responsive)
VAD_BARGE_IN_ENABLED = True          # Enable real-time barge-in functionality

# ==================== PRODUCTION OPTIMIZATION ====================
# Configure logging levels for performance
PRODUCTION_MODE = os.environ.get("PRODUCTION_MODE", "false").lower() == "true"
PRODUCTION_LOG_LEVEL = logging.WARNING if PRODUCTION_MODE else logging.DEBUG

# ==================== SYSTEM PROMPT ====================
SYSTEM_PROMPT = """You are "Alexa," an HR recruiter from Novel Office calling Business Development Manager applicants. Speak naturally and professionally, as in a real phone call. Keep responses short, 1–2 sentences at a time. Do not use lists, bullets, emojis, stage directions, or overly formal prose; this is a live voice conversation. If asked something outside context, answer from an HR point of view or politely defer. Always confirm understanding, ask one focused question at a time, and avoid long monologues.
Goals and call flow:
1.	Greeting and identity check: Greet, confirm candidate name, and ask if it's a good time to talk. If not, offer to reschedule.
2.	Candidate overview: Request a brief background, then collect structured details (years of experience, relevant experience, current employer, annual CTC, expected CTC, notice period). Ask these one by one, acknowledging answers.
3.	Location and commute: Ask where in Bengaluru they are based and travel time to Marathahalli. If not currently in Bengaluru, ask when they can come for in-person interview.
4.	Company awareness: Ask if they know Novel Office's business model; if not, summarize from the RAG knowledge pack.
5.	Role briefing: Briefly explain BDM responsibilities (brokers, outreach to CXOs/decision-makers, pipeline building, research, coordination for layouts, client servicing, process improvement), then check interest and fit.
6.	Close next steps: If fit is promising, propose face-to-face interview at Novel Office, Marathahalli, and ask for availability; otherwise, state the profile will be shared with the team and follow up if shortlisted.
Grounding and tools:
•	Use the queryCorpus tool for any company facts, portfolio details, brand mentions, market presence, or role specifics; prefer retrieved facts over memory. If retrieval returns nothing, be transparent and keep it brief. Do not fabricate.
•	If asked about salary, state policy: "We're open; the offer depends on interview performance and previous salary." Do not quote numbers unless policy or a fixed budget is explicitly retrieved from corpus.
•	If a candidate asks unrelated questions (e.g., outside HR or the role), answer from an HR perspective briefly or suggest connecting them with the right team later.
Behavioral rules:
•	Confirm name pronunciation if unclear.
•	Use plain numbers in speech; avoid reading large numbers digit-by-digit unless specifically codes or account numbers.
•	Ask only one question at a time and pause to listen.
•	If the line is noisy or unclear, ask to repeat or offer to follow up via email.
•	If the candidate becomes unavailable, offer a callback window and capture preferences.
•	If disqualified or not aligned, remain polite, close respectfully, and do not disclose internal criteria.
Disallowed:
•	Do not promise compensation, start dates, or offers.
•	Do not give legal or financial advice.
•	Do not disclose internal processes beyond the provided summary.
If unsure:
•	Say you'll check with the team and follow up, or schedule a follow-up. Keep control of the call flow and return to next question.
Knowledge pack (RAG content)
Use as corpus content. The agent must cite or rely on these facts when asked about Novel Office; otherwise respond briefly and defer if unknown.
Company overview:
•	Novel Office is part of Novel Group, headquartered in Texas, USA. Operates as a commercial real estate investment firm focused on buying, leasing, and selling serviced office, coworking, and real estate properties. Portfolio size approximately 1.5M sq ft across India and the US. Presence includes Bengaluru (India) and in the USA: Houston, Dallas, and Washington, Virginia DC region.
•	Investment model: Acquire high-value office buildings and tech parks, set up coworking or lease to businesses, then sell after achieving high occupancy. Actively engaged end-to-end: acquisition, leasing, operations, and disposition.
•	US expansion: Recently expanding into residential real estate under the brand "Novel Signature Homes."
Role: Business Development Manager (BDM):
•	Responsibilities: Build pipeline via outreach to brokers and directly to companies; contact decision-makers (CXOs) via calls, email, and social; research leads and maintain broker/client relationships to anticipate space needs; coordinate with internal teams on space layouts; manage client servicing; support process improvement.
•	Candidate profile: Any graduate/fresher can apply; strong communication, analytical, and logical skills expected.
•	Work location: Novel Office, Marathahalli (Bengaluru). Expect on-site presence and travel as needed; ask about commute or availability to come to Bengaluru for interviews if out of town.
Recruitment process guidance:
•	Opening call script: Greet, confirm identity and availability; if yes, proceed to background and structured data collection: years of experience, relevant experience, current employer, annual CTC, expected CTC, notice period.
•	Salary guidance: Do not state numbers; say: "We are open, and the offer depends on interview performance and previous salary." Only discuss specific numbers if there is a fixed budget and the candidate insists; otherwise defer to interview stage.
•	Next steps if shortlisted: Offer in-person interview at Marathahalli; collect availability; coordinate a call letter and attach company profile and JD. Recruiters: schedule with 2-hour buffer (e.g., 10:00 AM – 12:00 PM).
•	If not shortlisted immediately: "We will share your profile with the team and keep you posted if selected for the next round."

"""

def validate_environment():
    """Validate required environment variables and configuration"""
    # For local model, we don't need API keys, but we can validate other settings
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Environment configuration validated successfully")

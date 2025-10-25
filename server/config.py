"""
config.py
Configuration module for the Ultravox WebSocket server.
Contains all configuration constants, settings, and default values.
"""

import torch
import logging

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model configuration
ULTRAVOX_MODEL = 'fixie-ai/ultravox-v0_5-llama-3_2-1b'
PIPER_MODEL_NAME = "en_US-lessac-medium"
PIPER_ONNX_FILE = "en_US-lessac-medium.onnx"
PIPER_JSON_FILE = "en_US-lessac-medium.json"

# Server configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
MAX_MESSAGE_SIZE = 10_000_000  # 10MB
PING_INTERVAL = 30  # seconds
PING_TIMEOUT = 300  # seconds
CLOSE_TIMEOUT = 10  # seconds

# Audio configuration
CHUNK_SIZE = 256 * 1024  # 256 KB chunks for WebSocket streaming
AUDIO_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 22050
MAX_TTS_TEXT_LENGTH = 1000

# Threading configuration
MAX_WORKERS = 4
THREAD_NAME_PREFIX = "ultravox_inference"

# Logging configuration
LOG_LEVEL = logging.INFO

# Default conversation turns for HR recruiter role
DEFAULT_CONVERSATION_TURNS = [
    {
        "role": "system",
        "content": '''
        You are "Alexa," an HR recruiter from Novel Office calling Business Development Manager applicants. Speak naturally and professionally, as in a real phone call. Keep responses short, 1–2 sentences at a time. Do not use lists, bullets, emojis, stage directions, or overly formal prose; this is a live voice conversation. If asked something outside context, answer from an HR point of view or politely defer. Always confirm understanding, ask one focused question at a time, and avoid long monologues.
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
Always be more and more precise, and use less tokens to talk to users.
        '''
    }
]

DEFAULT_CONVERSATION_TURNS1 = [
    {
        "role": "system",
        "content": '''
                You are a helpful assistant that transcribes audio to text. Please transcribe the provided audio accurately.
        '''
    }
]

# Available TTS voices
AVAILABLE_VOICES = ["default", "multilingual", "indian", "us", "uk"]

# ==================== TRANSCRIPTION QUALITY CONFIGURATION ====================
# These parameters help prevent false transcriptions from background noise

# Minimum RMS energy for transcription (higher = stricter)
# Default: 1500 (much higher than VAD_ENERGY_THRESHOLD to avoid false positives)
TRANSCRIPTION_MIN_ENERGY = 1500

# Minimum ratio of speech-like samples required (0.0 to 1.0)
# Default: 0.25 (25% of samples must look like speech - much stricter)
TRANSCRIPTION_MIN_SPEECH_RATIO = 0.25

# Minimum audio duration to consider for transcription (milliseconds)
# Very short audio clips are more likely to be noise
TRANSCRIPTION_MIN_DURATION_MS = 300  # 300ms minimum

# Known false positive words to filter out
# These are common Whisper hallucinations for background noise
TRANSCRIPTION_FALSE_POSITIVES = [
    "you", "thank you", "thanks", "bye", "goodbye",
    "okay", "ok", "yeah", "yes", "no", "um", "uh"
]

# Maximum length for false positive filtering (characters)
# If transcription is shorter than this AND matches false positive, reject it
TRANSCRIPTION_FALSE_POSITIVE_MAX_LENGTH = 50

# Enable detailed transcription validation logging
TRANSCRIPTION_DEBUG_LOGGING = True

# ============================================================================
# THREE-TIER ADAPTIVE VALIDATION STRATEGY
# ============================================================================

# TIER 1: RELAXED for WebSocket validation gate (let borderline audio through)
# These are intentionally loose to avoid rejecting natural speech
UNIFIED_AUDIO_VALIDATION = True
UNIFIED_MIN_ENERGY = 500          # Much lower - allow very quiet speakers
UNIFIED_MIN_SPEECH_RATIO = 0.15    # Much lower - allow very slow speakers

# TIER 2: MEDIUM for Ultravox processing (moderate filter)
# If audio passes gate but is borderline, Ultravox should be cautious
ULTRAVOX_MIN_ENERGY = 800          # Lower for slow speech
ULTRAVOX_MIN_SPEECH_RATIO = 0.25    # Lower for slow speech

# TIER 3: STRICT for Whisper logging (prevent hallucinations)
# Whisper is prone to false positives on garbage audio
TRANSCRIPTION_MIN_ENERGY = 1200     # Keep high - prevent Whisper hallucinations
TRANSCRIPTION_MIN_SPEECH_RATIO = 0.35

# Static greeting configuration
STATIC_GREETING_ENABLED = True
STATIC_GREETING_MESSAGE = "Hello, I'm Alexa, an HR recruiter from Novel Office calling Business Development Manager applicants. Is this a good time to talk?"

# ==================== WHISPER CONFIGURATION ====================
# Whisper model language settings

# Force Whisper to transcribe only in English (disable auto-detection)
WHISPER_LANGUAGE = "en"  # Set to "en" for English only, None for auto-detection
WHISPER_TASK = "transcribe"  # "transcribe" or "translate"

# ==================== VAD (Voice Activity Detection) Configuration ====================
# UltraVAD configuration for interruption detection using transformers

# Enable/disable VAD functionality
VAD_ENABLED = False

# VAD model configuration
VAD_MODEL_NAME = "fixie-ai/ultraVAD"  # UltraVAD model from Hugging Face
VAD_SAMPLE_RATE = 16000  # Must match audio processing sample rate

# Speech detection thresholds
VAD_THRESHOLD = 0.6  # Speech probability threshold (0.5-0.7 balanced)
VAD_MIN_SPEECH_DURATION_MS = 300  # Minimum speech duration to trigger interruption
VAD_SPEECH_PAD_MS = 100  # Padding before/after speech segments

# Real-time processing configuration
VAD_FRAME_SIZE_MS = 30  # VAD processing frame size (30/60/90ms for real-time)
VAD_TEMPORAL_CONSISTENCY_FRAMES = 3  # Required consecutive speech frames for interruption

# Interruption detection settings
VAD_INTERRUPTION_COOLDOWN_MS = 500  # Minimum time between interruption detections
VAD_BACKGROUND_MONITORING_ENABLED = True  # Monitor for speech during AI TTS playback

# Debug and logging
VAD_DEBUG_LOGGING = True  # Enable detailed VAD logging
VAD_STATS_LOGGING = False  # Log audio statistics for debugging

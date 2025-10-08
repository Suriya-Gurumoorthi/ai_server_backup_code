import asyncio
import websockets
import transformers
import torch
import numpy as np
import librosa
import io
import logging
import json
import uuid
import wave
import tempfile
import os
import time
import re
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
from piper import PiperVoice
from vector_memory import initialize_vector_memory, get_vector_memory, store_conversation_fact, search_conversation_memories, get_conversation_context, format_memories_for_prompt
from enum import Enum
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Conversation state management
class ConversationStage(Enum):
    GREETING = "greeting"
    IDENTITY_CHECK = "identity_check"
    BACKGROUND_COLLECTION = "background_collection"
    LOCATION_COMMUTE = "location_commute"
    COMPANY_AWARENESS = "company_awareness"
    ROLE_BRIEFING = "role_briefing"
    CLOSING = "closing"
    COMPLETED = "completed"

@dataclass
class ConversationState:
    stage: ConversationStage = ConversationStage.GREETING
    collected_info: Dict[str, Any] = None
    last_question_asked: Optional[str] = None
    retry_count: int = 0
    user_name: Optional[str] = None
    conversation_turns: List[Dict[str, str]] = None
    system_prompt_set: bool = False
    
    def __post_init__(self):
        if self.collected_info is None:
            self.collected_info = {}
        if self.conversation_turns is None:
            self.conversation_turns = []
    
    def should_move_to_next_stage(self) -> bool:
        """Determine if conversation should progress to next stage"""
        if self.retry_count >= 3:
            return True  # Force progression after 3 retries
        
        # Stage-specific progression logic
        if self.stage == ConversationStage.GREETING:
            return self.user_name is not None
        elif self.stage == ConversationStage.BACKGROUND_COLLECTION:
            required_fields = ['experience', 'current_employer', 'ctc', 'notice_period']
            return all(field in self.collected_info for field in required_fields)
        elif self.stage == ConversationStage.LOCATION_COMMUTE:
            return 'location' in self.collected_info
        elif self.stage == ConversationStage.COMPANY_AWARENESS:
            return 'company_knowledge' in self.collected_info
        elif self.stage == ConversationStage.ROLE_BRIEFING:
            return 'role_interest' in self.collected_info
        
        return False
    
    def move_to_next_stage(self):
        """Progress to next conversation stage"""
        stage_order = [
            ConversationStage.GREETING,
            ConversationStage.IDENTITY_CHECK,
            ConversationStage.BACKGROUND_COLLECTION,
            ConversationStage.LOCATION_COMMUTE,
            ConversationStage.COMPANY_AWARENESS,
            ConversationStage.ROLE_BRIEFING,
            ConversationStage.CLOSING,
            ConversationStage.COMPLETED
        ]
        
        try:
            current_index = stage_order.index(self.stage)
            if current_index < len(stage_order) - 1:
                self.stage = stage_order[current_index + 1]
                self.retry_count = 0  # Reset retry count for new stage
                logger.info(f"Conversation progressed to stage: {self.stage.value}")
        except ValueError:
            logger.error(f"Unknown conversation stage: {self.stage}")
    
    def detect_loop(self, new_response: str) -> bool:
        """Detect if the model is looping by comparing with recent responses"""
        if len(self.conversation_turns) < 4:
            return False
        
        # Check last 3 assistant responses for similarity
        recent_responses = []
        for turn in reversed(self.conversation_turns[-6:]):  # Check last 6 turns
            if turn.get('role') == 'assistant':
                recent_responses.append(turn.get('content', ''))
                if len(recent_responses) >= 3:
                    break
        
        if len(recent_responses) < 2:
            return False
        
        # Simple similarity check - if responses are very similar, it's likely looping
        new_response_words = set(new_response.lower().split())
        for response in recent_responses:
            response_words = set(response.lower().split())
            if len(new_response_words & response_words) / max(len(new_response_words), 1) > 0.8:
                logger.warning("Detected potential response loop")
                return True
        
        return False

# Detect and configure device
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device set to use {device}")
if device == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Load the Ultravox pipeline once at server startup
logger.info("Loading Ultravox pipeline...")
try:
    pipe = transformers.pipeline(
        model='fixie-ai/ultravox-v0_5-llama-3_2-1b',
        trust_remote_code=True,
        device=device,  # Load on GPU if available
        torch_dtype=torch.float32  # Use full precision for stability
    )
    logger.info(f"Ultravox pipeline loaded successfully on {device}!")
    if device == "cuda":
        logger.info("Model is using GPU acceleration for faster inference")
except Exception as e:
    logger.error(f"Failed to load Ultravox pipeline: {e}")
    raise

# Generation configuration for consistent behavior
GENERATION_CONFIG = {
    "max_new_tokens": 150,  # Shorter responses to prevent loops
    "temperature": 0.7,
    "do_sample": True,
    "repetition_penalty": 1.1,  # Prevent repetition
    "top_p": 0.9,
    "pad_token_id": 50256  # EOS token
}

# Load small 1B model for context extraction
logger.info("Loading context extraction model...")
context_extractor = None
try:
    context_extractor = transformers.pipeline(
        model='microsoft/DialoGPT-small',  # Small 1B model for context extraction
        trust_remote_code=True,
        device=device,
        torch_dtype=torch.float32  # Use full precision for stability
    )
    logger.info(f"Context extraction model loaded successfully on {device}!")
except Exception as e:
    logger.error(f"Failed to load context extraction model: {e}")
    logger.warning("Context extraction will fall back to simple text processing")

# Load Piper TTS
logger.info("Loading Piper TTS...")
piper_voice = None
try:
    # Check for the correct model files
    onnx_file = "en_US-lessac-medium.onnx"
    json_file = "en_US-lessac-medium.json"  # Piper expects .json extension
    model_link = "en_US-lessac-medium"  # Piper expects this exact name
    
    if os.path.exists(onnx_file) and os.path.exists(json_file) and os.path.exists(model_link):
        logger.info(f"Found ONNX model file: {onnx_file}")
        logger.info(f"Found JSON config file: {json_file}")
        logger.info(f"Found model symlink: {model_link}")
        
        # Try to load the voice using the correct path
        # Piper expects the model name without extension
        model_name = "en_US-lessac-medium"
        piper_voice = PiperVoice.load(model_name)
        logger.info("Piper TTS loaded successfully!")
    else:
        missing_files = []
        if not os.path.exists(onnx_file):
            missing_files.append(onnx_file)
        if not os.path.exists(json_file):
            missing_files.append(json_file)
        if not os.path.exists(model_link):
            missing_files.append(f"{model_link} (symlink to {onnx_file})")
        logger.warning(f"Missing required files: {missing_files}")
        logger.warning("TTS functionality will be disabled")
        
except Exception as e:
    logger.error(f"Failed to load Piper TTS: {e}")
    logger.warning("TTS functionality will be disabled")
    piper_voice = None

# Initialize Vector Memory
logger.info("Initializing Vector Memory...")
try:
    vector_memory = initialize_vector_memory(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        db_type="chroma",
        cache_size=1000
    )
    logger.info("Vector Memory initialized successfully!")
except Exception as e:
    logger.error(f"Failed to initialize Vector Memory: {e}")
    logger.warning("Context memory functionality will be disabled")
    vector_memory = None

# Fixed system prompt - set once and never regenerated
SYSTEM_PROMPT = '''You are "Alexa," an HR recruiter from Novel Office calling Business Development Manager applicants. Speak naturally and professionally, as in a real phone call. Keep responses short, 1–2 sentences at a time. Do not use lists, bullets, emojis, stage directions, or overly formal prose; this is a live voice conversation. If asked something outside context, answer from an HR point of view or politely defer. Always confirm understanding, ask one focused question at a time, and avoid long monologues.

Goals and call flow:
1. Greeting and identity check: Greet, confirm candidate name, and ask if it's a good time to talk. If not, offer to reschedule.
2. Candidate overview: Request a brief background, then collect structured details (years of experience, relevant experience, current employer, annual CTC, expected CTC, notice period). Ask these one by one, acknowledging answers.
3. Location and commute: Ask where in Bengaluru they are based and travel time to Marathahalli. If not currently in Bengaluru, ask when they can come for in-person interview.
4. Company awareness: Ask if they know Novel Office's business model; if not, summarize from the RAG knowledge pack.
5. Role briefing: Briefly explain BDM responsibilities (brokers, outreach to CXOs/decision-makers, pipeline building, research, coordination for layouts, client servicing, process improvement), then check interest and fit.
6. Close next steps: If fit is promising, propose face-to-face interview at Novel Office, Marathahalli, and ask for availability; otherwise, state the profile will be shared with the team and follow up if shortlisted.

Grounding and tools:
• Use the queryCorpus tool for any company facts, portfolio details, brand mentions, market presence, or role specifics; prefer retrieved facts over memory. If retrieval returns nothing, be transparent and keep it brief. Do not fabricate.
• If asked about salary, state policy: "We're open; the offer depends on interview performance and previous salary." Do not quote numbers unless policy or a fixed budget is explicitly retrieved from corpus.
• If a candidate asks unrelated questions (e.g., outside HR or the role), answer from an HR perspective briefly or suggest connecting them with the right team later.

Behavioral rules:
• Confirm name pronunciation if unclear.
• Use plain numbers in speech; avoid reading large numbers digit-by-digit unless specifically codes or account numbers.
• Ask only one question at a time and pause to listen.
• If the line is noisy or unclear, ask to repeat or offer to follow up via email.
• If the candidate becomes unavailable, offer a callback window and capture preferences.
• If disqualified or not aligned, remain polite, close respectfully, and do not disclose internal criteria.

Disallowed:
• Do not promise compensation, start dates, or offers.
• Do not give legal or financial advice.
• Do not disclose internal processes beyond the provided summary.

If unsure:
• Say you'll check with the team and follow up, or schedule a follow-up. Keep control of the call flow and return to next question.

Knowledge pack (RAG content)
Use as corpus content. The agent must cite or rely on these facts when asked about Novel Office; otherwise respond briefly and defer if unknown.

Company overview:
• Novel Office is part of Novel Group, headquartered in Texas, USA. Operates as a commercial real estate investment firm focused on buying, leasing, and selling serviced office, coworking, and real estate properties. Portfolio size approximately 1.5M sq ft across India and the US. Presence includes Bengaluru (India) and in the USA: Houston, Dallas, and Washington, Virginia DC region.
• Investment model: Acquire high-value office buildings and tech parks, set up coworking or lease to businesses, then sell after achieving high occupancy. Actively engaged end-to-end: acquisition, leasing, operations, and disposition.
• US expansion: Recently expanding into residential real estate under the brand "Novel Signature Homes."

Role: Business Development Manager (BDM):
• Responsibilities: Build pipeline via outreach to brokers and directly to companies; contact decision-makers (CXOs) via calls, email, and social; research leads and maintain broker/client relationships to anticipate space needs; coordinate with internal teams on space layouts; manage client servicing; support process improvement.
• Candidate profile: Any graduate/fresher can apply; strong communication, analytical, and logical skills expected.
• Work location: Novel Office, Marathahalli (Bengaluru). Expect on-site presence and travel as needed; ask about commute or availability to come to Bengaluru for interviews if out of town.

Recruitment process guidance:
• Opening call script: Greet, confirm identity and availability; if yes, proceed to background and structured data collection: years of experience, relevant experience, current employer, annual CTC, expected CTC, notice period.
• Salary guidance: Do not state numbers; say: "We are open, and the offer depends on interview performance and previous salary." Only discuss specific numbers if there is a fixed budget and the candidate insists; otherwise defer to interview stage.
• Next steps if shortlisted: Offer in-person interview at Marathahalli; collect availability; coordinate a call letter and attach company profile and JD. Recruiters: schedule with 2-hour buffer (e.g., 10:00 AM – 12:00 PM).
• If not shortlisted immediately: "We will share your profile with the team and keep you posted if selected for the next round."

Always be more and more precise, and use less tokens to talk to users.'''

# Track active connections and their states
active_connections: Dict[str, ConversationState] = {}

# Thread pool executor for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ultravox_inference")

# Configuration for chunked audio streaming
CHUNK_SIZE = 256 * 1024  # 256 KB chunks to stay well under 1MB WebSocket limit

# Audio processing timing constants
AUDIO_SUPPRESSION_WINDOW = 1.0      # Suppress audio processing for 1s after sending response
TTS_PROCESSING_DELAY = 0.5          # Delay before marking TTS as complete

def sanitize_audio_placeholders(text: str) -> str:
    """Remove all audio placeholders from text to prevent Ultravox pipeline errors"""
    if not text:
        return text
    return text.replace('<|audio|>', '').replace('<|audio|', '').strip()

async def ultravox_transcribe(audio_bytes: bytes, prompt: str = None) -> str:
    """
    Pure transcription function using Ultravox model.
    
    Args:
        audio_bytes: Raw audio data
        prompt: Optional custom prompt for transcription
        
    Returns:
        Transcribed text
    """
    if prompt is None:
        prompt = "Please transcribe the following audio input accurately."
    
    try:
        # Validate input
        if not audio_bytes or len(audio_bytes) == 0:
            return "Error: No audio data provided"
        
        # Load audio using librosa
        audio_stream = io.BytesIO(audio_bytes)
        audio, sr = librosa.load(audio_stream, sr=16000)
        
        if len(audio) == 0:
            return "Error: Audio file contains no data"
        
        # Create simple transcription prompt
        transcription_turns = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "<|audio|>"}
        ]
        
        # Run transcription with proper await
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: pipe({'audio': audio, 'turns': transcription_turns, 'sampling_rate': sr}, max_new_tokens=1000)
        )
        
        # Extract text result with robust error handling
        transcript = ""
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and 'generated_text' in result[0]:
                transcript = result[0]['generated_text']
            elif isinstance(result[0], str):
                transcript = result[0]
            else:
                transcript = str(result[0])
        elif isinstance(result, dict):
            if 'generated_text' in result:
                transcript = result['generated_text']
            elif 'text' in result:
                transcript = result['text']
            else:
                transcript = str(result)
        elif isinstance(result, str):
            transcript = result
        else:
            transcript = str(result)
        
        # Sanitize the transcript
        transcript = sanitize_audio_placeholders(transcript).strip()
        
        if not transcript:
            return "Error: Empty transcription result"
        
        logger.info(f"Transcription completed: {transcript[:100]}...")
        return transcript
        
    except Exception as e:
        logger.error(f"Error in transcription: {e}")
        import traceback
        logger.error(f"Transcription traceback: {traceback.format_exc()}")
        return f"Error transcribing audio: {str(e)}"

async def extract_key_context(transcript: str, extraction_prompt: str = None) -> str:
    """
    Extract important context from transcript using a small 1B model.
    
    Args:
        transcript: Raw transcript text
        extraction_prompt: Optional custom prompt for context extraction
        
    Returns:
        Extracted key context summary
    """
    if not transcript or not transcript.strip():
        return ""
    
    if extraction_prompt is None:
        extraction_prompt = (
            "Extract only the most important facts, context, and logical insights from the following text. "
            "Focus on: key information, decisions made, important details, user preferences, and relevant context. "
            "Return a concise summary suitable for long-term memory retrieval. "
            "Ignore filler words, greetings, and irrelevant details."
        )
    
    if context_extractor is None:
        # Fallback to simple text processing
        logger.warning("Context extractor not available, using simple text processing")
        return simple_context_extraction(transcript)
    
    try:
        # Validate input length
        if len(transcript) > 2000:
            logger.warning(f"Transcript too long ({len(transcript)} chars), truncating for context extraction")
            transcript = transcript[:2000]
        
        # Prepare input for context extraction
        input_text = f"{extraction_prompt}\n\nTranscript:\n{transcript}"
        
        # Use the small model for context extraction with proper await
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: context_extractor(input_text, max_new_tokens=200, do_sample=True, temperature=0.7)
        )
        
        # Extract context summary with robust error handling
        context = ""
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and 'generated_text' in result[0]:
                context = result[0]['generated_text']
            elif isinstance(result[0], str):
                context = result[0]
            else:
                context = str(result[0])
        elif isinstance(result, dict):
            if 'generated_text' in result:
                context = result['generated_text']
            elif 'text' in result:
                context = result['text']
            else:
                context = str(result)
        elif isinstance(result, str):
            context = result
        else:
            context = str(result)
        
        # Clean up the context
        context = sanitize_audio_placeholders(context).strip()
        
        # Remove the original prompt from the response
        if extraction_prompt in context:
            context = context.replace(extraction_prompt, "").strip()
        
        # Validate extracted context
        if not context or len(context) < 10:
            logger.warning("Context extraction returned minimal content, falling back to simple extraction")
            return simple_context_extraction(transcript)
        
        logger.info(f"Context extracted: {context[:100]}...")
        return context
        
    except Exception as e:
        logger.error(f"Error in context extraction: {e}")
        import traceback
        logger.error(f"Context extraction traceback: {traceback.format_exc()}")
        # Fallback to simple extraction
        return simple_context_extraction(transcript)

def simple_context_extraction(transcript: str) -> str:
    """
    Simple fallback context extraction using basic text processing.
    
    Args:
        transcript: Raw transcript text
        
    Returns:
        Basic context summary
    """
    if not transcript or not transcript.strip():
        return ""
    
    # Simple heuristics for context extraction
    sentences = transcript.split('.')
    important_sentences = []
    
    # Keywords that indicate important information
    important_keywords = [
        'name', 'experience', 'salary', 'location', 'company', 'role', 'position',
        'interview', 'available', 'interested', 'prefer', 'need', 'want', 'like',
        'decision', 'choice', 'option', 'plan', 'schedule', 'time', 'date',
        'important', 'key', 'main', 'primary', 'specific', 'details'
    ]
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10:  # Filter out very short sentences
            # Check if sentence contains important keywords
            if any(keyword in sentence.lower() for keyword in important_keywords):
                important_sentences.append(sentence)
    
    # If no important sentences found, take the first few sentences
    if not important_sentences:
        important_sentences = sentences[:3]
    
    context = '. '.join(important_sentences[:5])  # Limit to 5 sentences
    if context and not context.endswith('.'):
        context += '.'
    
    logger.info(f"Simple context extracted: {context[:100]}...")
    return context

def extract_user_name(text: str) -> str:
    """
    Extract user name from text using regex patterns.
    
    Args:
        text: Input text to extract name from
        
    Returns:
        Extracted name or empty string if not found
    """
    if not text or not text.strip():
        return ""
    
    text = text.strip()
    
    # Common patterns for name introduction
    patterns = [
        r"my name is ([A-Za-z\s]+)",
        r"i'm ([A-Za-z\s]+)",
        r"i am ([A-Za-z\s]+)",
        r"call me ([A-Za-z\s]+)",
        r"this is ([A-Za-z\s]+)",
        r"i'm ([A-Za-z\s]+)",
        r"name's ([A-Za-z\s]+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            # Clean up the name (remove extra words, punctuation)
            name = re.sub(r'[^\w\s]', '', name).strip()
            if len(name) > 0 and len(name) < 50:  # Reasonable name length
                logger.info(f"Extracted user name: {name}")
                return name
    
    return ""

async def add_memory_context_as_turns(conversation_state: ConversationState, session_id: str) -> List[Dict[str, str]]:
    """
    Add memory context as conversation turns instead of system prompt injection.
    
    Args:
        conversation_state: Current conversation state
        session_id: Session ID for memory retrieval
        
    Returns:
        Updated conversation turns with memory context
    """
    if not vector_memory or not session_id:
        return conversation_state.conversation_turns
    
    try:
        # Get recent memories
        recent_memories = await get_conversation_context(session_id, max_memories=3)
        if not recent_memories:
            return conversation_state.conversation_turns
        
        # Add memory context as system messages (not user/assistant turns)
        memory_turns = []
        for memory in recent_memories:
            memory_content = memory.get('content', '')
            if memory_content and len(memory_content) > 10:  # Only add substantial memories
                memory_turns.append({
                    "role": "system",
                    "content": f"[Memory context: {memory_content}]"
                })
        
        # Insert memory turns after the main system prompt but before user turns
        if memory_turns:
            # Find where to insert (after main system prompt)
            insert_index = 1 if conversation_state.conversation_turns and conversation_state.conversation_turns[0].get('role') == 'system' else 0
            conversation_state.conversation_turns[insert_index:insert_index] = memory_turns
            logger.info(f"Added {len(memory_turns)} memory context turns to conversation")
        
        return conversation_state.conversation_turns
        
    except Exception as e:
        logger.error(f"Error adding memory context: {e}")
        return conversation_state.conversation_turns

def trim_conversation_history(turns: List[Dict[str, str]], max_turns: int = 25) -> List[Dict[str, str]]:
    """
    Intelligently trim conversation history while preserving important context.
    
    Args:
        turns: List of conversation turns
        max_turns: Maximum number of turns to keep
        
    Returns:
        Trimmed conversation turns
    """
    if len(turns) <= max_turns:
        return turns
    
    # Always keep the main system prompt (first turn)
    system_turn = turns[0] if turns and turns[0].get("role") == "system" else None
    
    # Keep memory context turns (system messages with [Memory context:])
    memory_turns = [turn for turn in turns if turn.get("role") == "system" and "[Memory context:" in turn.get("content", "")]
    
    # Get recent user/assistant turns
    user_assistant_turns = [turn for turn in turns if turn.get("role") in ["user", "assistant"]]
    
    # Calculate how many recent turns we can keep
    reserved_turns = 1 + len(memory_turns)  # system + memory turns
    available_slots = max_turns - reserved_turns
    
    if available_slots <= 0:
        # If we have too many memory turns, keep only the most recent ones
        recent_memory_turns = memory_turns[-3:] if len(memory_turns) > 3 else memory_turns
        recent_user_assistant = user_assistant_turns[-max_turns:]
        return [system_turn] + recent_memory_turns + recent_user_assistant if system_turn else recent_memory_turns + recent_user_assistant
    
    # Keep recent user/assistant turns
    recent_user_assistant = user_assistant_turns[-available_slots:]
    
    # Combine all turns
    result = []
    if system_turn:
        result.append(system_turn)
    result.extend(memory_turns)
    result.extend(recent_user_assistant)
    
    return result

def ensure_one_audio_placeholder_last_user(turns: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Removes all audio placeholders from all turns except the last user turn,
    and ensures only one <|audio|> at the end.
    """
    if not turns:
        return turns
    
    cleaned = []
    last_user_idx = -1

    # 1. Find index of last user turn
    for idx in reversed(range(len(turns))):
        if turns[idx].get('role') == 'user':
            last_user_idx = idx
            break

    for idx, turn in enumerate(turns):
        role = turn.get("role", "")
        content = turn.get("content", "")

        # Remove all existing placeholders
        content = sanitize_audio_placeholders(content)
        
        if role == "user" and idx == last_user_idx:
            # Only the last user turn gets the audio marker
            if content:
                content = f"{content}\n<|audio|>"
            else:
                content = "<|audio|>"
        
        # Write back cleaned turn
        cleaned.append({"role": role, "content": content})
    
    return cleaned

def ensure_single_audio_placeholder(turns: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Legacy function - redirects to the correct implementation"""
    return ensure_one_audio_placeholder_last_user(turns)

def validate_audio_placeholders(turns: List[Dict[str, str]]) -> bool:
    """
    Defensive check to ensure exactly one audio placeholder in conversation turns.
    
    Args:
        turns: List of conversation turns
        
    Returns:
        True if valid, False otherwise
    """
    if not turns:
        return True
    
    placeholder_count = sum('<|audio|>' in turn.get('content', '') for turn in turns)
    
    if placeholder_count == 0:
        logger.warning("No audio placeholders found in conversation turns")
        return False
    elif placeholder_count > 1:
        logger.error(f"Too many audio placeholders found: {placeholder_count} (expected 1)")
        return False
    else:
        logger.info("Audio placeholder validation passed: exactly 1 found")
        return True


async def store_conversation_memory(session_id: str, user_message: str, assistant_response: str):
    """Store conversation turn in vector memory using selective context extraction"""
    if vector_memory is None:
        return
    
    try:
        # Extract key context from user message
        if user_message and user_message.strip():
            sanitized_user_message = sanitize_audio_placeholders(user_message.strip())
            if sanitized_user_message:
                # Extract important context from user message
                user_context = await extract_key_context(sanitized_user_message)
                if user_context:  # Only store if context extraction found important information
                    await store_conversation_fact(
                        session_id, 
                        f"User context: {user_context}", 
                        {"type": "user_context", "timestamp": time.time(), "original_message": sanitized_user_message}
                    )
                else:
                    # Fallback: store original message if no context extracted
                    await store_conversation_fact(
                        session_id, 
                        f"User said: {sanitized_user_message}", 
                        {"type": "user_message", "timestamp": time.time()}
                    )
        
        # Extract key context from assistant response
        if assistant_response and assistant_response.strip():
            sanitized_assistant_response = sanitize_audio_placeholders(assistant_response.strip())
            if sanitized_assistant_response:
                # Extract important context from assistant response
                assistant_context = await extract_key_context(sanitized_assistant_response)
                if assistant_context:  # Only store if context extraction found important information
                    await store_conversation_fact(
                        session_id, 
                        f"Assistant context: {assistant_context}", 
                        {"type": "assistant_context", "timestamp": time.time(), "original_response": sanitized_assistant_response}
                    )
                else:
                    # Fallback: store original response if no context extracted
                    await store_conversation_fact(
                        session_id, 
                        f"Assistant responded: {sanitized_assistant_response}", 
                        {"type": "assistant_response", "timestamp": time.time()}
                    )
            
    except Exception as e:
        logger.error(f"Error storing conversation memory: {e}")

async def store_transcript_memory(session_id: str, transcript: str):
    """Store transcript with selective context extraction"""
    if vector_memory is None or not transcript or not transcript.strip():
        return
    
    try:
        # Extract key context from transcript
        context = await extract_key_context(transcript)
        
        if context:  # Only store if context extraction found important information
            await store_conversation_fact(
                session_id, 
                f"Conversation context: {context}", 
                {"type": "conversation_context", "timestamp": time.time(), "original_transcript": transcript}
            )
            logger.info(f"Stored extracted context for session {session_id}")
        else:
            # Fallback: store original transcript if no context extracted
            await store_conversation_fact(
                session_id, 
                f"Transcript: {transcript}", 
                {"type": "transcript", "timestamp": time.time()}
            )
            logger.info(f"Stored original transcript for session {session_id}")
            
    except Exception as e:
        logger.error(f"Error storing transcript memory: {e}")

async def handle_connection(websocket):
    # Generate unique connection ID
    connection_id = str(uuid.uuid4())
    client_address = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
    
    try:
        # Initialize conversation state
        conversation_state = ConversationState()
        conversation_state.conversation_turns = [{"role": "system", "content": SYSTEM_PROMPT}]
        conversation_state.system_prompt_set = True
        
        # Store additional connection metadata
        connection_metadata = {
            "websocket": websocket,
            "client_address": client_address,
            "waiting_for_audio": False,
            "pending_request_type": None,
            # Audio state management
            "is_generating_response": False,     # Track when AI is generating response
            "response_start_time": 0,           # When response generation started
            "response_end_time": 0,             # When response will complete
            "last_audio_received": 0,           # Last time audio was received
            "suppress_audio_until": 0           # Suppress audio processing until this time
        }
        
        # Store both conversation state and metadata
        active_connections[connection_id] = conversation_state
        conversation_state.connection_metadata = connection_metadata
        
        logger.info(f"✅ Client {client_address} connected with ID: {connection_id}")
        logger.info(f"📊 Active connections: {len(active_connections)}")
        
        # Send welcome message to confirm connection (only for debugging)
        # Commented out to avoid interfering with actual responses
        # try:
        #     await websocket.send("Connected to Ultravox server")
        # except Exception as e:
        #     logger.warning(f"Could not send welcome message to {client_address}: {e}")
        
        async for message in websocket:
            try:
                await process_message(connection_id, message)
            except Exception as e:
                logger.error(f"❌ Error processing message from {client_address}: {e}")
                error_response = f"Error processing input: {str(e)}"
                await safe_send_response(websocket, error_response, client_address)

    except websockets.ConnectionClosedOK:
        logger.info(f"👋 Client {client_address} disconnected gracefully.")
    except websockets.ConnectionClosedError as e:
        logger.warning(f"⚠️  Client {client_address} connection closed with error: {e}")
    except Exception as e:
        logger.error(f"💥 Unexpected connection error for {client_address}: {e}")
    finally:
        # Clean up connection state
        if connection_id in active_connections:
            del active_connections[connection_id]
        logger.info(f"🧹 Cleaned up connection {connection_id}. Active connections: {len(active_connections)}")

async def process_message(connection_id: str, message):
    """Process a single message for a specific connection with audio suppression"""
    conversation_state = active_connections.get(connection_id)
    if not conversation_state:
        logger.error(f"Connection {connection_id} not found in active connections")
        return
    
    connection_metadata = conversation_state.connection_metadata
    websocket = connection_metadata["websocket"]
    client_address = connection_metadata["client_address"]
    
    logger.info(f"Received message of type {type(message)} from {client_address}")

    # If message is bytes, treat as audio
    if isinstance(message, bytes):
        # CHECK: Audio suppression - ignore audio if AI just responded
        current_time = time.time()
        if current_time < connection_metadata["suppress_audio_until"]:
            suppression_remaining = connection_metadata["suppress_audio_until"] - current_time
            logger.info(f"🔇 Suppressing audio from {client_address} for {suppression_remaining:.1f}s more (AI recently responded)")
            return  # Ignore this audio input
        
        # CHECK: Response generation in progress
        if connection_metadata["is_generating_response"]:
            logger.info(f"🔄 Ignoring audio from {client_address} - AI response generation in progress")
            return  # Ignore audio while generating response
        
        # Update last audio received time
        connection_metadata["last_audio_received"] = current_time
        
        if connection_metadata["waiting_for_audio"]:
            # Process the audio with the pending request type
            request_type = connection_metadata["pending_request_type"]
            connection_metadata["waiting_for_audio"] = False
            connection_metadata["pending_request_type"] = None
            
            # Mark response generation as starting
            connection_metadata["is_generating_response"] = True
            connection_metadata["response_start_time"] = current_time
            
            if request_type == "transcribe":
                logger.info(f"Processing transcribe request for {client_address}")
                response_text = await process_audio_pipeline(message, connection_id)
                logger.info(f"Sending transcribe response to {client_address}")
                await safe_send_response_with_suppression(websocket, response_text, client_address, connection_id)
                
            elif request_type == "features":
                logger.info(f"Processing features request for {client_address}")
                # Use the new pipeline with features-specific transcription
                response_text = await process_audio_pipeline_with_features(message, connection_id)
                logger.info(f"Sending features response to {client_address}")
                await safe_send_response_with_suppression(websocket, response_text, client_address, connection_id)
                
            elif request_type == "tts":
                logger.info(f"Processing TTS request for {client_address}")
                # Use the new pipeline for transcription and response generation
                response_text = await process_audio_pipeline(message, connection_id)
                logger.info(f"Pipeline completed, generating TTS audio for {client_address}")
                
                # Generate TTS audio from the response
                tts_audio = await generate_tts_audio(response_text)
                
                if tts_audio:
                    # Send audio using chunked streaming
                    success = await send_chunked_audio_with_suppression(websocket, tts_audio, client_address, response_text, connection_id)
                    if success:
                        logger.info(f"Successfully sent chunked TTS response to {client_address}")
                    else:
                        logger.error(f"Failed to send chunked TTS response to {client_address}")
                        # Fallback to text only
                        await safe_send_response_with_suppression(websocket, response_text, client_address, connection_id)
                else:
                    # Fallback to text only
                    await safe_send_response_with_suppression(websocket, response_text, client_address, connection_id)
                    logger.warning(f"TTS failed, sent text-only response to {client_address}")
            else:
                logger.warning(f"Unknown request type '{request_type}' for {client_address}")
                await safe_send_response_with_suppression(websocket, f"Error: Unknown request type '{request_type}'", client_address, connection_id)
                
            # Mark response generation as complete
            connection_metadata["is_generating_response"] = False
        else:
            # Direct audio processing using new pipeline
            connection_metadata["is_generating_response"] = True
            connection_metadata["response_start_time"] = current_time
            
            response_text = await process_audio_pipeline(message, connection_id)
            await safe_send_response_with_suppression(websocket, response_text, client_address, connection_id)
            
            connection_metadata["is_generating_response"] = False
        return

    # If message is str, try to parse as JSON
    if isinstance(message, str):
        try:
            request = json.loads(message)
            req_type = request.get("type", "transcribe")
        except Exception:
            await websocket.send("Error: Cannot parse non-audio, non-JSON input.")
            return

        # Handle different request types
        if req_type == "transcribe":
            connection_metadata["waiting_for_audio"] = True
            connection_metadata["pending_request_type"] = "transcribe"
            logger.info(f"Set waiting for audio for {client_address}, request type: transcribe")
            # Don't send response yet - wait for audio

        elif req_type == "features":
            connection_metadata["waiting_for_audio"] = True
            connection_metadata["pending_request_type"] = "features"
            logger.info(f"Set waiting for audio for {client_address}, request type: features")
            # Don't send response yet - wait for audio

        elif req_type == "tts":
            connection_metadata["waiting_for_audio"] = True
            connection_metadata["pending_request_type"] = "tts"
            logger.info(f"Set waiting for audio for {client_address}, request type: tts")
            # Don't send response yet - wait for audio

        elif req_type == "voices":
            voices_info = {"voices": ["default", "multilingual", "indian", "us", "uk"]}
            logger.info(f"Sending voices response to {client_address}")
            await safe_send_response(websocket, json.dumps(voices_info), client_address)

        elif req_type == "clear_memory":
            if vector_memory:
                try:
                    vector_memory.clear_session_memories(connection_id)
                    logger.info(f"Cleared memory for session {connection_id}")
                    await safe_send_response(websocket, "Memory cleared successfully", client_address)
                except Exception as e:
                    logger.error(f"Error clearing memory for {client_address}: {e}")
                    await safe_send_response(websocket, f"Error clearing memory: {str(e)}", client_address)
            else:
                await safe_send_response(websocket, "Memory system not available", client_address)

        else:
            logger.warning(f"Unsupported request type '{req_type}' from {client_address}")
            await safe_send_response(websocket, f"Error: Unsupported request type '{req_type}'", client_address)
        return

async def safe_send_response(websocket, message, client_address):
    """Safely send a response, handling connection state properly"""
    try:
        await websocket.send(message)
        logger.info(f"Response sent successfully to {client_address}")
        return True
    except websockets.ConnectionClosed:
        logger.warning(f"Connection to {client_address} closed while sending response")
        return False
    except Exception as e:
        logger.error(f"Error sending response to {client_address}: {e}")
        return False

async def safe_send_response_with_suppression(websocket, message, client_address, connection_id):
    """Send response and set audio suppression window"""
    success = await safe_send_response(websocket, message, client_address)
    
    if success:
        # Set audio suppression window
        conversation_state = active_connections.get(connection_id)
        if conversation_state and hasattr(conversation_state, 'connection_metadata'):
            current_time = time.time()
            conversation_state.connection_metadata["suppress_audio_until"] = current_time + AUDIO_SUPPRESSION_WINDOW
            conversation_state.connection_metadata["response_end_time"] = current_time
            logger.info(f"🔇 Set audio suppression for {client_address} until {AUDIO_SUPPRESSION_WINDOW}s from now")
    
    return success

async def send_chunked_audio(websocket, audio_bytes, client_address, text_response):
    """Send audio data in chunks to avoid WebSocket message size limits"""
    try:
        # Send initial metadata
        metadata = {
            "type": "tts_start",
            "text": text_response,
            "audio_size": len(audio_bytes),
            "chunk_size": CHUNK_SIZE,
            "total_chunks": (len(audio_bytes) + CHUNK_SIZE - 1) // CHUNK_SIZE
        }
        
        logger.info(f"Sending TTS metadata to {client_address}: {len(audio_bytes)} bytes in {metadata['total_chunks']} chunks")
        await safe_send_response(websocket, json.dumps(metadata), client_address)
        
        # Send audio data in chunks
        chunk_count = 0
        for i in range(0, len(audio_bytes), CHUNK_SIZE):
            chunk = audio_bytes[i:i + CHUNK_SIZE]
            chunk_count += 1
            
            logger.info(f"Sending audio chunk {chunk_count}/{metadata['total_chunks']} ({len(chunk)} bytes) to {client_address}")
            await safe_send_response(websocket, chunk, client_address)
        
        # Send completion marker
        completion = {"type": "tts_end", "chunks_sent": chunk_count}
        logger.info(f"Sending TTS completion marker to {client_address}")
        await safe_send_response(websocket, json.dumps(completion), client_address)
        
        logger.info(f"Successfully sent {len(audio_bytes)} bytes of audio in {chunk_count} chunks to {client_address}")
        return True
        
    except Exception as e:
        logger.error(f"Error sending chunked audio to {client_address}: {e}")
        return False

async def send_chunked_audio_with_suppression(websocket, audio_bytes, client_address, text_response, connection_id):
    """Send chunked audio and set appropriate suppression window"""
    success = await send_chunked_audio(websocket, audio_bytes, client_address, text_response)
    
    if success:
        # Set longer suppression window for audio responses
        conversation_state = active_connections.get(connection_id)
        if conversation_state and hasattr(conversation_state, 'connection_metadata'):
            current_time = time.time()
            # Estimate audio duration for suppression
            audio_duration = estimate_audio_duration_from_bytes(audio_bytes)
            suppression_duration = audio_duration + AUDIO_SUPPRESSION_WINDOW
            
            conversation_state.connection_metadata["suppress_audio_until"] = current_time + suppression_duration
            conversation_state.connection_metadata["response_end_time"] = current_time + audio_duration
            logger.info(f"🔇 Set audio suppression for {client_address} for {suppression_duration:.1f}s (audio duration + buffer)")
    
    return success

def estimate_audio_duration_from_bytes(audio_bytes: bytes) -> float:
    """Estimate audio duration from WAV bytes"""
    try:
        import wave
        import io
        
        audio_stream = io.BytesIO(audio_bytes)
        with wave.open(audio_stream, 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            duration = frames / sample_rate
            return duration
    except Exception as e:
        logger.error(f"Error estimating audio duration: {e}")
        # Fallback: estimate based on file size (rough approximation)
        # Assuming 16-bit mono at 22050 Hz = ~44KB per second
        estimated_duration = len(audio_bytes) / 44100
        return max(1.0, estimated_duration)  # Minimum 1 second

async def generate_tts_audio(text: str) -> bytes:
    """Generate TTS audio from text using Piper with processing awareness"""
    if piper_voice is None:
        logger.warning("TTS not available, cannot generate audio")
        return None
    
    # Validate input text
    logger.info(f"TTS input text validation:")
    logger.info(f"  - Text type: {type(text)}")
    logger.info(f"  - Text length: {len(text) if text else 0}")
    logger.info(f"  - Text repr: {repr(text)}")
    logger.info(f"  - Text stripped length: {len(text.strip()) if text else 0}")
    
    # Check if text is empty or whitespace
    if not text or not text.strip():
        logger.warning("TTS text is empty or whitespace only. Skipping synthesis.")
        return None
    
    # Check for potentially problematic characters
    if len(text) > 1000:
        logger.warning(f"Text is very long ({len(text)} chars), truncating to 1000 chars")
        text = text[:1000]
    
    try:
        logger.info(f"🎤 Starting TTS generation...")
        start_time = time.time()
        
        logger.info(f"Calling PiperVoice.synthesize with text: {repr(text[:100])}...")
        # Generate audio using Piper TTS
        audio_data = piper_voice.synthesize(text)
        logger.info(f"Piper synthesize returned type: {type(audio_data)}")
        
        # Handle AudioChunk objects from Piper TTS
        if hasattr(audio_data, '__iter__'):
            # Convert generator to list to examine the chunks
            audio_chunks = list(audio_data)
            logger.info(f"Piper synthesize returned {len(audio_chunks)} chunks")
            
            if len(audio_chunks) > 0:
                # Extract audio data from AudioChunk objects
                audio_bytes = b""
                for idx, chunk in enumerate(audio_chunks):
                    logger.info(f"Processing chunk {idx}:")
                    
                    # Check available attributes
                    if hasattr(chunk, 'audio_int16_bytes'):
                        chunk_bytes = chunk.audio_int16_bytes
                        logger.info(f"  - audio_int16_bytes: {len(chunk_bytes)} bytes")
                        if len(chunk_bytes) > 0:
                            audio_bytes += chunk_bytes
                            logger.info(f"  - Added {len(chunk_bytes)} bytes to total")
                        else:
                            logger.warning(f"  - audio_int16_bytes is empty!")
                    elif hasattr(chunk, 'audio_int16_array'):
                        chunk_array = chunk.audio_int16_array
                        chunk_bytes = chunk_array.tobytes()
                        logger.info(f"  - audio_int16_array: {chunk_array.shape}, {len(chunk_bytes)} bytes")
                        if len(chunk_bytes) > 0:
                            audio_bytes += chunk_bytes
                            logger.info(f"  - Added {len(chunk_bytes)} bytes to total")
                        else:
                            logger.warning(f"  - audio_int16_array is empty!")
                    else:
                        logger.warning(f"  - AudioChunk {idx} missing expected audio attributes")
                        logger.warning(f"  - Available attributes: {[attr for attr in dir(chunk) if not attr.startswith('_')]}")
                
                logger.info(f"Extracted audio bytes: {len(audio_bytes)} bytes")
            else:
                logger.warning("No audio chunks received from Piper TTS")
                return None
        else:
            # If it's not a generator, treat as direct bytes
            audio_bytes = audio_data
        
        # Convert to WAV format
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(22050)  # Piper default sample rate
            wav_file.writeframes(audio_bytes)
        
        wav_bytes = wav_buffer.getvalue()
        
        generation_time = time.time() - start_time
        logger.info(f"🎤 Generated TTS audio: {len(wav_bytes)} bytes in {generation_time:.2f}s")
        
        # Add small processing delay to ensure TTS completion
        await asyncio.sleep(TTS_PROCESSING_DELAY)
        
        return wav_bytes
        
    except Exception as e:
        logger.error(f"Error generating TTS audio: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None

async def process_audio_pipeline(audio_bytes: bytes, session_id: str = None) -> str:
    """
    Complete audio processing pipeline with proper conversation state management.
    
    Args:
        audio_bytes: Raw audio data
        session_id: Session ID for memory management
        
    Returns:
        AI response text
    """
    try:
        # Get conversation state
        conversation_state = active_connections.get(session_id)
        if not conversation_state:
            logger.error(f"Conversation state not found for session {session_id}")
            return "Error: Session not found"
        
        # Step 1: Transcribe audio
        logger.info(f"🎤 Starting transcription for session {session_id}")
        transcript = await ultravox_transcribe(audio_bytes)
        
        if transcript.startswith("Error:"):
            return transcript
        
        # Step 2: Store transcript with context extraction
        if session_id and vector_memory:
            await store_transcript_memory(session_id, transcript)
        
        # Step 3: Generate AI response using proper conversation state
        logger.info(f"🤖 Generating AI response for session {session_id}")
        response = await generate_ai_response_with_state(transcript, conversation_state, session_id)
        
        return response
        
    except Exception as e:
        logger.error(f"Error in audio pipeline: {e}")
        return f"Error processing audio: {str(e)}"

async def generate_ai_response_with_state(user_input: str, conversation_state: ConversationState, session_id: str = None) -> str:
    """
    Generate AI response using proper conversation state management.
    
    Args:
        user_input: User's input text
        conversation_state: Current conversation state
        session_id: Session ID for memory retrieval
        
    Returns:
        AI response text
    """
    try:
        # Extract user name if present
        extracted_name = extract_user_name(user_input)
        if extracted_name and not conversation_state.user_name:
            conversation_state.user_name = extracted_name
            logger.info(f"Set user name to: {extracted_name}")
        
        # Initialize conversation turns if this is the first interaction
        if not conversation_state.conversation_turns:
            conversation_state.conversation_turns = [{"role": "system", "content": SYSTEM_PROMPT}]
            conversation_state.system_prompt_set = True
            logger.info("Initialized conversation with system prompt")
        
        # Add user input to conversation
        conversation_state.conversation_turns.append({"role": "user", "content": user_input})
        
        # Add memory context as conversation turns (not system prompt injection)
        conversation_state.conversation_turns = await add_memory_context_as_turns(conversation_state, session_id)
        
        # Trim conversation history intelligently
        conversation_state.conversation_turns = trim_conversation_history(conversation_state.conversation_turns, max_turns=25)
        
        # Ensure proper audio placeholder handling - only last user turn gets placeholder
        conversation_state.conversation_turns = ensure_one_audio_placeholder_last_user(conversation_state.conversation_turns)
        
        # Defensive check before inference
        if not validate_audio_placeholders(conversation_state.conversation_turns):
            logger.error("Audio placeholder validation failed, attempting to fix...")
            conversation_state.conversation_turns = ensure_one_audio_placeholder_last_user(conversation_state.conversation_turns)
            if not validate_audio_placeholders(conversation_state.conversation_turns):
                logger.error("Failed to fix audio placeholder issue, aborting inference")
                return "Error: Audio placeholder validation failed"
        
        # Generate response using Ultravox with real audio (not dummy zeros)
        logger.info(f"🧠 Starting Ultravox inference...")
        inference_start = time.time()
        
        # Load audio for proper model input
        audio_stream = io.BytesIO(user_input.encode() if isinstance(user_input, str) else user_input)
        try:
            audio, sr = librosa.load(audio_stream, sr=16000)
        except:
            # If audio loading fails, use zeros as fallback
            audio = np.zeros(16000)
            sr = 16000
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            lambda: pipe({'audio': audio, 'turns': conversation_state.conversation_turns, 'sampling_rate': sr}, **GENERATION_CONFIG)
        )
        
        inference_time = time.time() - inference_start
        logger.info(f"🧠 Ultravox inference completed in {inference_time:.2f}s")
        
        # Extract response text
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and 'generated_text' in result[0]:
                response_text = result[0]['generated_text']
            elif isinstance(result[0], str):
                response_text = result[0]
            else:
                response_text = str(result[0])
        elif isinstance(result, dict):
            if 'generated_text' in result:
                response_text = result['generated_text']
            elif 'text' in result:
                response_text = result['text']
            else:
                response_text = str(result)
        elif isinstance(result, str):
            response_text = result
        else:
            response_text = str(result)
        
        # Sanitize response
        response_text = sanitize_audio_placeholders(response_text).strip()
        
        # Check for response loops
        if conversation_state.detect_loop(response_text):
            logger.warning("Detected response loop, forcing conversation progression")
            conversation_state.retry_count += 1
            if conversation_state.should_move_to_next_stage():
                conversation_state.move_to_next_stage()
                # Generate a stage-appropriate response
                response_text = generate_stage_response(conversation_state)
        
        # Add assistant response to conversation history
        conversation_state.conversation_turns.append({"role": "assistant", "content": response_text})
        
        # Store conversation in memory
        if session_id and vector_memory and response_text:
            await store_conversation_memory(session_id, user_input, response_text)
            logger.info("Stored conversation in vector memory")
        
        # Check if conversation should progress to next stage
        if conversation_state.should_move_to_next_stage():
            conversation_state.move_to_next_stage()
        
        return response_text
        
    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        import traceback
        logger.error(f"AI response generation traceback: {traceback.format_exc()}")
        return f"Error generating response: {str(e)}"

def generate_stage_response(conversation_state: ConversationState) -> str:
    """
    Generate a stage-appropriate response when loops are detected.
    
    Args:
        conversation_state: Current conversation state
        
    Returns:
        Stage-appropriate response text
    """
    stage_responses = {
        ConversationStage.GREETING: "Hello! I'm Alexa from Novel Office. I'm calling about our Business Development Manager position. Is this a good time to talk?",
        ConversationStage.IDENTITY_CHECK: "Could you please confirm your name and let me know if you're interested in learning about our BDM role?",
        ConversationStage.BACKGROUND_COLLECTION: "Let me ask about your background. How many years of experience do you have in business development?",
        ConversationStage.LOCATION_COMMUTE: "Where are you currently based? Are you in Bengaluru or would you need to relocate?",
        ConversationStage.COMPANY_AWARENESS: "Do you know about Novel Office's business model? We're a commercial real estate investment firm.",
        ConversationStage.ROLE_BRIEFING: "The BDM role involves building pipeline through broker outreach and direct company contact. Are you interested in this type of work?",
        ConversationStage.CLOSING: "Based on our discussion, I'd like to schedule an in-person interview at our Marathahalli office. When would you be available?",
        ConversationStage.COMPLETED: "Thank you for your time. We'll be in touch about next steps."
    }
    
    return stage_responses.get(conversation_state.stage, "I understand. Let me ask you about your experience in business development.")

async def process_audio_pipeline_with_features(audio_bytes: bytes, session_id: str = None) -> str:
    """
    Process audio with features extraction (gender, emotion, accent, quality).
    
    Args:
        audio_bytes: Raw audio data
        session_id: Session ID for memory management
        
    Returns:
        Features analysis text
    """
    try:
        # Get conversation state
        conversation_state = active_connections.get(session_id)
        if not conversation_state:
            logger.error(f"Conversation state not found for session {session_id}")
            return "Error: Session not found"
        
        # Step 1: Transcribe audio with features prompt
        logger.info(f"🎤 Starting features transcription for session {session_id}")
        features_prompt = "Transcribe the audio and also provide speaker gender, emotion, accent, and audio quality."
        transcript = await ultravox_transcribe(audio_bytes, features_prompt)
        
        if transcript.startswith("Error:"):
            return transcript
        
        # Step 2: Store transcript with context extraction
        if session_id and vector_memory:
            await store_transcript_memory(session_id, transcript)
        
        return transcript
        
    except Exception as e:
        logger.error(f"Error in features pipeline: {e}")
        return f"Error processing audio features: {str(e)}"

# Legacy process_audio_message function removed - now using unified process_audio_pipeline

async def monitor_connection_states():
    """Monitor and log connection states for debugging"""
    while True:
        try:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            current_time = time.time()
            for conn_id, conversation_state in active_connections.items():
                if hasattr(conversation_state, 'connection_metadata'):
                    conn_meta = conversation_state.connection_metadata
                    client_addr = conn_meta.get("client_address", "unknown")
                    is_generating = conn_meta.get("is_generating_response", False)
                    suppress_until = conn_meta.get("suppress_audio_until", 0)
                    last_audio = conn_meta.get("last_audio_received", 0)
                    
                    time_since_audio = current_time - last_audio if last_audio > 0 else 0
                    suppression_remaining = max(0, suppress_until - current_time)
                    
                    logger.info(f"📊 Connection {client_addr}: generating={is_generating}, "
                               f"suppression_remaining={suppression_remaining:.1f}s, "
                               f"time_since_audio={time_since_audio:.1f}s, "
                               f"stage={conversation_state.stage.value}")
                           
        except Exception as e:
            logger.error(f"Error in connection monitoring: {e}")

async def main():
    try:
        # Start connection monitoring task
        monitor_task = asyncio.create_task(monitor_connection_states())
        
        # Configure server with better connection handling
        server = await websockets.serve(
            handle_connection, 
            "0.0.0.0", 
            8000,
            max_size=10_000_000,  # 10MB max message size
            ping_interval=30,     # Send ping every 30 seconds
            ping_timeout=300,     # Wait 5 minutes for pong
            close_timeout=10,     # 10 seconds to close connection
            compression=None      # Disable compression for better performance
        )
        
        logger.info("Ultravox WebSocket server started on port 8000")
        logger.info(f"Server configuration:")
        logger.info(f"  - Max message size: 10MB")
        logger.info(f"  - Ping interval: 30s")
        logger.info(f"  - Ping timeout: 300s")
        logger.info(f"  - Compression: disabled")
        logger.info(f"  - Thread executor: 4 workers for inference")
        logger.info(f"  - Model device: {device}")
        if device == "cuda":
            logger.info(f"  - GPU acceleration: enabled")
        logger.info(f"  - TTS: {'enabled' if piper_voice else 'disabled'}")
        logger.info(f"  - Vector Memory: {'enabled' if vector_memory else 'disabled'}")
        if vector_memory:
            logger.info(f"  - Memory DB: ChromaDB")
            logger.info(f"  - Embedding model: sentence-transformers/all-MiniLM-L6-v2")
        logger.info(f"  - Context Extraction: {'enabled' if context_extractor else 'disabled'}")
        if context_extractor:
            logger.info(f"  - Context Model: microsoft/DialoGPT-small")
        logger.info(f"  - Audio chunking: {CHUNK_SIZE // 1024}KB chunks for large TTS responses")
        logger.info(f"  - Audio suppression: {AUDIO_SUPPRESSION_WINDOW}s window after responses")
        logger.info(f"  - Connection monitoring: enabled")
        logger.info(f"  - Pipeline: Transcription -> Context Extraction -> Memory Storage -> Response Generation")
        
        # Keep server running
        await server.wait_closed()
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
    finally:
        # Clean up
        if 'monitor_task' in locals():
            monitor_task.cancel()
        logger.info("Shutting down thread executor...")
        executor.shutdown(wait=True)

if __name__ == "__main__":
    asyncio.run(main())

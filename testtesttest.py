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
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
from piper import PiperVoice

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        torch_dtype=torch.float16 if device == "cuda" else torch.float32  # Use half precision on GPU for efficiency
    )
    logger.info(f"Ultravox pipeline loaded successfully on {device}!")
    if device == "cuda":
        logger.info("Model is using GPU acceleration for faster inference")
except Exception as e:
    logger.error(f"Failed to load Ultravox pipeline: {e}")
    raise

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

# Initial prompt for conversation style
default_turns = [
    {
        "role": "system",
        "content": '''
        You are “Alexa,” an HR recruiter from Novel Office calling Business Development Manager applicants. Speak naturally and professionally, as in a real phone call. Keep responses short, 1–2 sentences at a time. Do not use lists, bullets, emojis, stage directions, or overly formal prose; this is a live voice conversation. If asked something outside context, answer from an HR point of view or politely defer. Always confirm understanding, ask one focused question at a time, and avoid long monologues.
Goals and call flow:
1.	Greeting and identity check: Greet, confirm candidate name, and ask if it’s a good time to talk. If not, offer to reschedule.
2.	Candidate overview: Request a brief background, then collect structured details (years of experience, relevant experience, current employer, annual CTC, expected CTC, notice period). Ask these one by one, acknowledging answers.
3.	Location and commute: Ask where in Bengaluru they are based and travel time to Marathahalli. If not currently in Bengaluru, ask when they can come for in-person interview.
4.	Company awareness: Ask if they know Novel Office’s business model; if not, summarize from the RAG knowledge pack.
5.	Role briefing: Briefly explain BDM responsibilities (brokers, outreach to CXOs/decision-makers, pipeline building, research, coordination for layouts, client servicing, process improvement), then check interest and fit.
6.	Close next steps: If fit is promising, propose face-to-face interview at Novel Office, Marathahalli, and ask for availability; otherwise, state the profile will be shared with the team and follow up if shortlisted.
Grounding and tools:
•	Use the queryCorpus tool for any company facts, portfolio details, brand mentions, market presence, or role specifics; prefer retrieved facts over memory. If retrieval returns nothing, be transparent and keep it brief. Do not fabricate.
•	If asked about salary, state policy: “We’re open; the offer depends on interview performance and previous salary.” Do not quote numbers unless policy or a fixed budget is explicitly retrieved from corpus.
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
•	Say you’ll check with the team and follow up, or schedule a follow-up. Keep control of the call flow and return to next question.
Knowledge pack (RAG content)
Use as corpus content. The agent must cite or rely on these facts when asked about Novel Office; otherwise respond briefly and defer if unknown.
Company overview:
•	Novel Office is part of Novel Group, headquartered in Texas, USA. Operates as a commercial real estate investment firm focused on buying, leasing, and selling serviced office, coworking, and real estate properties. Portfolio size approximately 1.5M sq ft across India and the US. Presence includes Bengaluru (India) and in the USA: Houston, Dallas, and Washington, Virginia DC region.
•	Investment model: Acquire high-value office buildings and tech parks, set up coworking or lease to businesses, then sell after achieving high occupancy. Actively engaged end-to-end: acquisition, leasing, operations, and disposition.
•	US expansion: Recently expanding into residential real estate under the brand “Novel Signature Homes.”
Role: Business Development Manager (BDM):
•	Responsibilities: Build pipeline via outreach to brokers and directly to companies; contact decision-makers (CXOs) via calls, email, and social; research leads and maintain broker/client relationships to anticipate space needs; coordinate with internal teams on space layouts; manage client servicing; support process improvement.
•	Candidate profile: Any graduate/fresher can apply; strong communication, analytical, and logical skills expected.
•	Work location: Novel Office, Marathahalli (Bengaluru). Expect on-site presence and travel as needed; ask about commute or availability to come to Bengaluru for interviews if out of town.
Recruitment process guidance:
•	Opening call script: Greet, confirm identity and availability; if yes, proceed to background and structured data collection: years of experience, relevant experience, current employer, annual CTC, expected CTC, notice period.
•	Salary guidance: Do not state numbers; say: “We are open, and the offer depends on interview performance and previous salary.” Only discuss specific numbers if there is a fixed budget and the candidate insists; otherwise defer to interview stage.
•	Next steps if shortlisted: Offer in-person interview at Marathahalli; collect availability; coordinate a call letter and attach company profile and JD. Recruiters: schedule with 2-hour buffer (e.g., 10:00 AM – 12:00 PM).
•	If not shortlisted immediately: “We will share your profile with the team and keep you posted if selected for the next round.”
Always be more and more precise, and use less tokens to talk to users.
        '''
    }
    
]

# Track active connections and their states
active_connections: Dict[str, Dict[str, Any]] = {}

# Thread pool executor for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ultravox_inference")

# Configuration for chunked audio streaming
CHUNK_SIZE = 256 * 1024  # 256 KB chunks to stay well under 1MB WebSocket limit

async def handle_connection(websocket):
    # Generate unique connection ID
    connection_id = str(uuid.uuid4())
    client_address = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
    
    try:
        # Initialize connection state
        active_connections[connection_id] = {
            "websocket": websocket,
            "client_address": client_address,
            "conversation_turns": default_turns.copy(),
            "waiting_for_audio": False,
            "pending_request_type": None
        }
        
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
    """Process a single message for a specific connection"""
    connection = active_connections.get(connection_id)
    if not connection:
        logger.error(f"Connection {connection_id} not found in active connections")
        return
    
    websocket = connection["websocket"]
    client_address = connection["client_address"]
    
    logger.info(f"Received message of type {type(message)} from {client_address}")

    # If message is bytes, treat as audio
    if isinstance(message, bytes):
        if connection["waiting_for_audio"]:
            # Process the audio with the pending request type
            request_type = connection["pending_request_type"]
            connection["waiting_for_audio"] = False
            connection["pending_request_type"] = None
            
            if request_type == "transcribe":
                logger.info(f"Processing transcribe request for {client_address}")
                response_text = await process_audio_message(message, connection["conversation_turns"])
                logger.info(f"Sending transcribe response to {client_address}")
                await safe_send_response(websocket, response_text, client_address)
            elif request_type == "features":
                logger.info(f"Processing features request for {client_address}")
                features_prompt = (
                    "Transcribe the audio and also provide speaker gender, emotion, accent, and audio quality."
                )
                turns = [{"role": "system", "content": features_prompt}]
                response_text = await process_audio_message(message, turns)
                logger.info(f"Sending features response to {client_address}")
                await safe_send_response(websocket, response_text, client_address)
            elif request_type == "tts":
                logger.info(f"Processing TTS request for {client_address}")
                # First transcribe the audio
                response_text = await process_audio_message(message, connection["conversation_turns"])
                logger.info(f"Transcription completed, generating TTS audio for {client_address}")
                
                # Generate TTS audio from the transcription
                tts_audio = await generate_tts_audio(response_text)
                
                if tts_audio:
                    # Send audio using chunked streaming to avoid WebSocket size limits
                    success = await send_chunked_audio(websocket, tts_audio, client_address, response_text)
                    if success:
                        logger.info(f"Successfully sent chunked TTS response to {client_address}")
                    else:
                        logger.error(f"Failed to send chunked TTS response to {client_address}")
                        # Fallback to text only
                        await safe_send_response(websocket, response_text, client_address)
                else:
                    # Fallback to text only
                    await safe_send_response(websocket, response_text, client_address)
                    logger.warning(f"TTS failed, sent text-only response to {client_address}")
            else:
                logger.warning(f"Unknown request type '{request_type}' for {client_address}")
                await safe_send_response(websocket, f"Error: Unknown request type '{request_type}'", client_address)
        else:
            # Direct audio processing (legacy mode)
            response_text = await process_audio_message(message, connection["conversation_turns"])
            await safe_send_response(websocket, response_text, client_address)
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
            connection["waiting_for_audio"] = True
            connection["pending_request_type"] = "transcribe"
            logger.info(f"Set waiting for audio for {client_address}, request type: transcribe")
            # Don't send response yet - wait for audio

        elif req_type == "features":
            connection["waiting_for_audio"] = True
            connection["pending_request_type"] = "features"
            logger.info(f"Set waiting for audio for {client_address}, request type: features")
            # Don't send response yet - wait for audio

        elif req_type == "tts":
            connection["waiting_for_audio"] = True
            connection["pending_request_type"] = "tts"
            logger.info(f"Set waiting for audio for {client_address}, request type: tts")
            # Don't send response yet - wait for audio

        elif req_type == "voices":
            voices_info = {"voices": ["default", "multilingual", "indian", "us", "uk"]}
            logger.info(f"Sending voices response to {client_address}")
            await safe_send_response(websocket, json.dumps(voices_info), client_address)

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

async def generate_tts_audio(text: str) -> bytes:
    """Generate TTS audio from text using Piper"""
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
                # Examine the first chunk to understand its structure
                first_chunk = audio_chunks[0]
                logger.info(f"First chunk type: {type(first_chunk)}")
                logger.info(f"First chunk attributes: {dir(first_chunk)}")
                
                # Extract audio data from AudioChunk objects
                # Piper AudioChunk has audio_int16_bytes and audio_int16_array attributes
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
        logger.info(f"Generated TTS audio: {len(wav_bytes)} bytes")
        return wav_bytes
        
    except Exception as e:
        logger.error(f"Error generating TTS audio: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None

async def process_audio_message(audio_bytes, turns):
    try:
        # Load to numpy from bytes—assuming WAV
        audio_stream = io.BytesIO(audio_bytes)
        audio, sr = librosa.load(audio_stream, sr=16000)
        logger.info(f"Loaded audio: {len(audio)} samples at {sr}Hz")
        if len(audio) == 0:
            return "Error: Audio file contains no data"
    except Exception as e:
        logger.error(f"Failed to decode audio: {e}")
        return f"Error decoding audio: {str(e)}"

    # Pass audio and prompt turns to Ultravox (offloaded to thread executor)
    try:
        loop = asyncio.get_event_loop()
        # Offload blocking inference to thread executor to prevent handshake timeouts
        result = await loop.run_in_executor(
            executor, 
            lambda: pipe({'audio': audio, 'turns': turns, 'sampling_rate': sr}, max_new_tokens=1000)
        )
        logger.info("Ultravox inference completed in thread executor")
    except Exception as e:
        logger.error(f"Ultravox pipeline error: {e}")
        return f"Error processing audio with Ultravox: {str(e)}"

    # Extract text result with robust type checking
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
    logger.info(f"Ultravox reply: {response_text}")
    logger.info(f"Ultravox reply validation:")
    logger.info(f"  - Response type: {type(response_text)}")
    logger.info(f"  - Response length: {len(response_text) if response_text else 0}")
    logger.info(f"  - Response repr: {repr(response_text)}")
    logger.info(f"  - Response stripped length: {len(response_text.strip()) if response_text else 0}")
    return response_text

async def main():
    try:
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
        logger.info(f"  - Audio chunking: {CHUNK_SIZE // 1024}KB chunks for large TTS responses")
        
        # Keep server running
        await server.wait_closed()
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
    finally:
        # Clean up thread executor
        logger.info("Shutting down thread executor...")
        executor.shutdown(wait=True)

if __name__ == "__main__":
    asyncio.run(main())

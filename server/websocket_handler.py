"""
websocket_handler.py
WebSocket connection handler module.
Manages WebSocket connections, message processing, and client state.
"""

import json
import uuid
import logging
import asyncio
import websockets
from typing import Dict, Any

from config import (
    DEFAULT_CONVERSATION_TURNS, AVAILABLE_VOICES,
    TRANSCRIPTION_MIN_ENERGY, TRANSCRIPTION_MIN_SPEECH_RATIO,
    TRANSCRIPTION_FALSE_POSITIVES, TRANSCRIPTION_FALSE_POSITIVE_MAX_LENGTH,
    UNIFIED_AUDIO_VALIDATION, UNIFIED_MIN_ENERGY, UNIFIED_MIN_SPEECH_RATIO,
    STATIC_GREETING_ENABLED, STATIC_GREETING_MESSAGE
)
from utils import safe_send_response, get_connection_info, should_transcribe_audio, unified_audio_validation
from models import model_manager
from audio_processor import audio_processor
from transcription_manager import transcription_manager
from prompt_logger import prompt_logger


class WebSocketHandler:
    """Handles WebSocket connections and message processing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_connections: Dict[str, Dict[str, Any]] = {}
    
    def _should_process_audio(self, audio_bytes: bytes) -> bool:
        """
        Unified audio validation for all processing paths.
        Ensures Whisper and Ultravox are synchronized using the same strict criteria.
        """
        if not UNIFIED_AUDIO_VALIDATION:
            return True  # Skip validation if disabled
        
        # Use unified validation with the same strict criteria for both Whisper and Ultravox
        result = unified_audio_validation(
            audio_bytes, 
            min_energy=UNIFIED_MIN_ENERGY, 
            min_speech_ratio=UNIFIED_MIN_SPEECH_RATIO
        )
        
        # Debug logging
        self.logger.info(f"[UNIFIED] Audio validation result: {result} (energy={UNIFIED_MIN_ENERGY}, speech_ratio={UNIFIED_MIN_SPEECH_RATIO})")
        
        return result
    
    async def _send_static_greeting(self, connection_id: str, websocket, client_address: str):
        """Send the static greeting message immediately after connection."""
        try:
            if connection_id not in self.active_connections:
                self.logger.error(f"Connection {connection_id} not found for greeting")
                return
            
            connection = self.active_connections[connection_id]
            
            # Check if greeting has already been sent
            if connection.get("greeting_sent", False):
                self.logger.info(f"Greeting already sent for connection {connection_id}")
                return
            
            self.logger.info(f"🎤 Sending static greeting to {client_address}: {STATIC_GREETING_MESSAGE}")
            
            # Generate TTS audio for the greeting message
            greeting_audio = await audio_processor.generate_tts_audio(STATIC_GREETING_MESSAGE)
            
            if greeting_audio:
                # Send the greeting audio directly to the client
                await websocket.send(greeting_audio)
                self.logger.info(f"✅ Static greeting audio sent to {client_address} ({len(greeting_audio)} bytes)")
            else:
                # Fallback: send text response if TTS fails
                self.logger.warning(f"TTS failed for greeting, sending text to {client_address}")
                await safe_send_response(websocket, STATIC_GREETING_MESSAGE, client_address)
            
            # Mark greeting as sent
            connection["greeting_sent"] = True
            
            # Add greeting to conversation history if session exists
            if connection.get("session_created", False):
                transcription_manager.add_ai_transcription(connection_id, STATIC_GREETING_MESSAGE)
                self.logger.info(f"Added greeting to conversation history for {connection_id}")
            
        except Exception as e:
            self.logger.error(f"Error sending static greeting to {client_address}: {e}")
            # Don't fail the connection if greeting fails
    
    def _ensure_session_created(self, connection_id: str) -> bool:
        """Ensure a session is created for the connection if it doesn't exist yet."""
        connection = self.active_connections.get(connection_id)
        if not connection:
            return False
        
        if not connection.get("session_created", False):
            # Create session only when first conversation message is received
            session_id = transcription_manager.create_session(connection_id)
            connection["session_id"] = session_id
            connection["session_created"] = True
            
            # Create prompt logging session
            prompt_logger.create_session(connection_id, session_id)
            
            self.logger.info(f"Created session {session_id} for connection {connection_id} on first message")
            return True
        
        return True
    
    async def handle_connection(self, websocket):
        """Handle a new WebSocket connection."""
        connection_id = str(uuid.uuid4())
        client_address = get_connection_info(websocket)
        
        try:
            # Initialize connection state without creating session yet
            # Session will be created only when first conversation message is received
            self.active_connections[connection_id] = {
                "websocket": websocket,
                "client_address": client_address,
                "conversation_turns": DEFAULT_CONVERSATION_TURNS.copy(),
                "waiting_for_audio": False,
                "pending_request_type": None,
                "session_id": None,  # Will be set when first message is processed
                "session_created": False,  # Track if session has been created
                "greeting_sent": False  # Track if static greeting has been sent
            }
            
            self.logger.info(f"✅ Client {client_address} connected with ID: {connection_id}")
            self.logger.info(f"📊 Active connections: {len(self.active_connections)}")
            
            # Send static greeting message immediately after connection
            if STATIC_GREETING_ENABLED:
                await self._send_static_greeting(connection_id, websocket, client_address)
            
            # Process messages from the client
            async for message in websocket:
                try:
                    await self.process_message(connection_id, message)
                except Exception as e:
                    self.logger.error(f"❌ Error processing message from {client_address}: {e}")
                    error_response = f"Error processing input: {str(e)}"
                    await safe_send_response(websocket, error_response, client_address)

        except websockets.ConnectionClosedOK:
            self.logger.info(f"👋 Client {client_address} disconnected gracefully.")
        except websockets.ConnectionClosedError as e:
            self.logger.warning(f"⚠️  Client {client_address} connection closed with error: {e}")
        except Exception as e:
            self.logger.error(f"💥 Unexpected connection error for {client_address}: {e}")
        finally:
            # Clean up connection state and end transcription session
            if connection_id in self.active_connections:
                connection = self.active_connections[connection_id]
                # Only end session if it was actually created
                if connection.get("session_created", False):
                    await transcription_manager.end_session(connection_id)
                    prompt_logger.end_session(connection_id)
                del self.active_connections[connection_id]
            self.logger.info(f"🧹 Cleaned up connection {connection_id}. Active connections: {len(self.active_connections)}")
    
    async def process_message(self, connection_id: str, message):
        """Process a single message for a specific connection."""
        connection = self.active_connections.get(connection_id)
        if not connection:
            self.logger.error(f"Connection {connection_id} not found in active connections")
            return
        
        websocket = connection["websocket"]
        client_address = connection["client_address"]
        
        self.logger.info(f"Received message of type {type(message)} from {client_address}")

        # Handle binary audio messages
        if isinstance(message, bytes):
            await self._handle_audio_message(connection, message)
            return

        # Handle text/JSON messages
        if isinstance(message, str):
            await self._handle_text_message(connection, message)
            return
    
    async def _handle_audio_message(self, connection: Dict[str, Any], audio_bytes: bytes):
        """Handle binary audio message with unified validation."""
        websocket = connection["websocket"]
        client_address = connection["client_address"]
        
        # Find the connection_id for this connection
        connection_id = None
        for conn_id, conn_data in self.active_connections.items():
            if conn_data == connection:
                connection_id = conn_id
                break
        
        if not connection_id:
            self.logger.error(f"Could not find connection ID for {client_address}")
            await safe_send_response(websocket, "Error: Connection not found", client_address)
            return
        
        # CRITICAL FIX: Apply unified validation FIRST, before any AI processing
        if not self._should_process_audio(audio_bytes):
            self.logger.info(f"Audio rejected at validation gate - no processing at all")
            # Don't process, don't log, don't add to history
            # Optionally send a user-facing message (optional)
            return
        
        # Only after validation passes, proceed with the rest
        if not self._ensure_session_created(connection_id):
            self.logger.error(f"Failed to create session for connection {connection_id}")
            await safe_send_response(websocket, "Error: Failed to create session", client_address)
            return
        
        if connection["waiting_for_audio"]:
            # Process the audio with the pending request type
            request_type = connection["pending_request_type"]
            connection["waiting_for_audio"] = False
            connection["pending_request_type"] = None
            
            if request_type == "transcribe":
                await self._process_transcribe_request(websocket, client_address, audio_bytes, connection)
            elif request_type == "features":
                await self._process_features_request(websocket, client_address, audio_bytes)
            elif request_type == "tts":
                await self._process_tts_request(websocket, client_address, audio_bytes, connection)
            else:
                self.logger.warning(f"Unknown request type '{request_type}' for {client_address}")
                await safe_send_response(websocket, f"Error: Unknown request type '{request_type}'", client_address)
        else:
            # Direct audio processing
            self.logger.info(f"[GATE] Audio passed gate validation - processing")
            
            conversation_context = transcription_manager.get_conversation_context_for_ai(connection_id)
            combined_turns = []
            
            if connection["conversation_turns"] and connection["conversation_turns"][0].get("role") == "system":
                combined_turns.append(connection["conversation_turns"][0])
            
            combined_turns.extend(conversation_context)
            
            # Call Ultravox - it will apply TIER 2 validation internally
            response_text = await model_manager.process_audio(audio_bytes, combined_turns, connection_id)
            
            # Check if response is a rejection marker
            if response_text.startswith("[AUDIO_REJECTED_BY_ULTRAVOX_VALIDATION]"):
                self.logger.info(f"[ULTRAVOX] Audio rejected after gate - borderline audio")
                # Don't send anything to user, don't log to history, just return
                return
            
            if response_text.startswith("["):
                # Other internal errors
                self.logger.error(f"[ULTRAVOX] Internal error: {response_text}")
                # Don't expose internal errors to user, don't log
                return
            
            self.logger.info(f"[ULTRAVOX] AI Response: {response_text[:100]}...")
            
            # Generate TTS
            tts_audio = await audio_processor.generate_tts_audio(response_text)
            
            if tts_audio:
                await websocket.send(tts_audio)
            else:
                await safe_send_response(websocket, response_text, client_address)
            
            # Add AI response to history ONLY if it's valid and not a marker
            transcription_manager.add_ai_transcription(connection_id, response_text)
            
            # Background transcription for logging
            async def log_transcription(audio, conn_id):
                try:
                    user_transcription = await model_manager.transcribe_audio(audio, conn_id)
                    
                    if user_transcription and len(user_transcription.strip()) > 0:
                        transcription_lower = user_transcription.strip().lower()
                        if transcription_lower not in TRANSCRIPTION_FALSE_POSITIVES:
                            transcription_manager.add_user_transcription(conn_id, user_transcription)
                            self.logger.info(f"[BACKGROUND] User transcription logged: {user_transcription[:50]}...")
                        else:
                            self.logger.info(f"[BACKGROUND] Rejected false positive: '{user_transcription}'")
                except Exception as e:
                    self.logger.error(f"[BACKGROUND] Transcription error: {e}")
            
            asyncio.create_task(log_transcription(audio_bytes, connection_id))
    
    async def _handle_text_message(self, connection: Dict[str, Any], message: str):
        """Handle text/JSON message."""
        websocket = connection["websocket"]
        client_address = connection["client_address"]
        
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
            self.logger.info(f"Set waiting for audio for {client_address}, request type: transcribe")

        elif req_type == "features":
            connection["waiting_for_audio"] = True
            connection["pending_request_type"] = "features"
            self.logger.info(f"Set waiting for audio for {client_address}, request type: features")

        elif req_type == "tts":
            connection["waiting_for_audio"] = True
            connection["pending_request_type"] = "tts"
            self.logger.info(f"Set waiting for audio for {client_address}, request type: tts")

        elif req_type == "voices":
            voices_info = {"voices": AVAILABLE_VOICES}
            self.logger.info(f"Sending voices response to {client_address}")
            await safe_send_response(websocket, json.dumps(voices_info), client_address)

        else:
            self.logger.warning(f"Unsupported request type '{req_type}' from {client_address}")
            await safe_send_response(websocket, f"Error: Unsupported request type '{req_type}'", client_address)
    
    def _detect_and_update_candidate_name(self, connection_id: str, user_transcription: str):
        """Detect if user provided their name and update session state."""
        import re
        
        # Common patterns for name introduction
        name_patterns = [
            r"my name is (\w+)",
            r"i am (\w+)",
            r"i'm (\w+)",
            r"call me (\w+)",
            r"this is (\w+)",
            r"(\w+) here",
            r"(\w+) speaking"
        ]
        
        user_text = user_transcription.lower().strip()
        
        for pattern in name_patterns:
            match = re.search(pattern, user_text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Filter out common false positives
                if name not in ["alex", "alexa", "rachel", "emily", "richard", "sir", "madam", "yes", "no", "okay", "ok"]:
                    transcription_manager.update_candidate_name(connection_id, name)
                    self.logger.info(f"Detected candidate name: {name} for connection {connection_id}")
                    break
    
    async def _process_transcribe_request(self, websocket, client_address: str, audio_bytes: bytes, connection: Dict[str, Any]):
        """Process transcribe request with parallel processing and audio validation."""
        connection_id = None
        for conn_id, conn_data in self.active_connections.items():
            if conn_data == connection:
                connection_id = conn_id
                break
        
        if not connection_id:
            self.logger.error(f"Could not find connection ID for {client_address}")
            await safe_send_response(websocket, "Error: Connection not found", client_address)
            return
        
        self.logger.info(f"Processing transcribe request for {client_address}")
        
        # VALIDATION: Check if audio is worth transcribing (unified validation)
        if not self._should_process_audio(audio_bytes):
            self.logger.info(f"Rejecting transcribe request - audio appears to be noise/silence")
            await safe_send_response(websocket, "[No speech detected]", client_address)
            return
        
        # -- Critical Path: Transcribe user input first, then generate AI response --
        self.logger.info(f"Processing audio for AI response for {client_address}")
        
        # STEP 1: Transcribe user input first (this is critical for proper context)
        self.logger.info(f"Transcribing user input for {client_address}")
        user_transcription = await model_manager.transcribe_audio(audio_bytes, connection_id)
        
        # Validate transcription result
        if not user_transcription or len(user_transcription.strip()) == 0:
            self.logger.warning(f"Empty transcription result, treating as silence")
            await safe_send_response(websocket, "[No speech detected]", client_address)
            return
        
        # Check for common false positives
        transcription_lower = user_transcription.strip().lower()
        if transcription_lower in TRANSCRIPTION_FALSE_POSITIVES and len(audio_bytes) < 8000:
            # Likely false positive if transcription is short AND audio is brief
            self.logger.warning(f"Likely false positive detected: '{user_transcription}' (matched: {transcription_lower}) - rejecting")
            await safe_send_response(websocket, "[Background noise filtered]", client_address)
            return
        
        transcription_manager.add_user_transcription(connection_id, user_transcription)
        
        # Check if user provided their name and update session state
        self._detect_and_update_candidate_name(connection_id, user_transcription)
        
        self.logger.info(f"User transcription: {user_transcription[:50]}...")
        
        # STEP 2: Get updated conversation context (now includes current user input)
        conversation_context = transcription_manager.get_conversation_context_for_ai(connection_id)
        
        # Combine with default system prompt
        combined_turns = connection["conversation_turns"].copy()
        combined_turns.extend(conversation_context)
        
        # STEP 3: Generate AI response with proper context
        response_text = await model_manager.process_audio(audio_bytes, combined_turns, connection_id)
        self.logger.info(f"AI Response: {response_text[:100]}...")
        
        # STEP 4: Send response to user
        await safe_send_response(websocket, response_text, client_address)
        
        # STEP 5: Add AI response to history
        transcription_manager.add_ai_transcription(connection_id, response_text)
    
    async def _process_features_request(self, websocket, client_address: str, audio_bytes: bytes):
        """Process features request with audio validation."""
        self.logger.info(f"Processing features request for {client_address}")
        
        # VALIDATION: Check if audio is worth processing (unified validation)
        if not self._should_process_audio(audio_bytes):
            self.logger.info(f"Rejecting features request - audio appears to be noise/silence")
            await safe_send_response(websocket, "[No speech detected - unable to analyze features]", client_address)
            return
        
        features_prompt = "Transcribe the audio and also provide speaker gender, emotion, accent, and audio quality."
        turns = [{"role": "system", "content": features_prompt}]
        response_text = await model_manager.process_audio(audio_bytes, turns)
        self.logger.info(f"Sending features response to {client_address}")
        await safe_send_response(websocket, response_text, client_address)
    
    async def _process_tts_request(self, websocket, client_address: str, audio_bytes: bytes, connection: Dict[str, Any]):
        """Process TTS request with parallel processing and audio validation."""
        connection_id = None
        for conn_id, conn_data in self.active_connections.items():
            if conn_data == connection:
                connection_id = conn_id
                break
        
        if not connection_id:
            self.logger.error(f"Could not find connection ID for {client_address}")
            await safe_send_response(websocket, "Error: Connection not found", client_address)
            return
        
        self.logger.info(f"Processing TTS request for {client_address}")
        
        # VALIDATION: Check if audio is worth transcribing (unified validation)
        if not self._should_process_audio(audio_bytes):
            self.logger.info(f"Rejecting TTS request - audio appears to be noise/silence")
            # Generate a brief "I didn't catch that" response
            response_text = "I didn't catch that. Could you please repeat?"
            tts_audio = await audio_processor.generate_tts_audio(response_text)
            if tts_audio:
                await websocket.send(tts_audio)
            else:
                await safe_send_response(websocket, response_text, client_address)
            return
        
        # -- Critical Path: Transcribe user input first, then generate AI response --
        self.logger.info(f"Processing audio for AI response for {client_address}")
        
        # STEP 1: Transcribe user input first
        self.logger.info(f"Transcribing user input for {client_address}")
        user_transcription = await model_manager.transcribe_audio(audio_bytes, connection_id)
        
        # Validate transcription result
        if not user_transcription or len(user_transcription.strip()) == 0:
            self.logger.warning(f"Empty transcription result")
            response_text = "I didn't catch that. Could you please repeat?"
            tts_audio = await audio_processor.generate_tts_audio(response_text)
            if tts_audio:
                await websocket.send(tts_audio)
            else:
                await safe_send_response(websocket, response_text, client_address)
            return
        
        # Check for common false positives
        transcription_lower = user_transcription.strip().lower()
        if transcription_lower in TRANSCRIPTION_FALSE_POSITIVES and len(audio_bytes) < 8000:
            self.logger.warning(f"Likely false positive: '{user_transcription}' (matched: {transcription_lower}) - requesting clarification")
            response_text = "I didn't quite catch that. Could you please speak a bit louder?"
            tts_audio = await audio_processor.generate_tts_audio(response_text)
            if tts_audio:
                await websocket.send(tts_audio)
            else:
                await safe_send_response(websocket, response_text, client_address)
            return
        
        transcription_manager.add_user_transcription(connection_id, user_transcription)
        
        # Check if user provided their name
        self._detect_and_update_candidate_name(connection_id, user_transcription)
        
        self.logger.info(f"User transcription: {user_transcription[:50]}...")
        
        # STEP 2-5: Continue with normal processing...
        conversation_context = transcription_manager.get_conversation_context_for_ai(connection_id)
        combined_turns = connection["conversation_turns"].copy()
        combined_turns.extend(conversation_context)
        
        response_text = await model_manager.process_audio(audio_bytes, combined_turns, connection_id)
        self.logger.info(f"AI Response: {response_text[:100]}...")
        
        transcription_manager.add_ai_transcription(connection_id, response_text)
        
        # Generate TTS audio
        self.logger.info(f"Generating TTS audio for {client_address}")
        tts_audio = await audio_processor.generate_tts_audio(response_text)
        
        if tts_audio:
            await websocket.send(tts_audio)
            self.logger.info(f"Successfully sent TTS audio to {client_address}")
        else:
            await safe_send_response(websocket, response_text, client_address)


# Global WebSocket handler instance
websocket_handler = WebSocketHandler()

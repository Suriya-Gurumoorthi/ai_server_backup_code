"""
WebSocket connection handler module.
Manages WebSocket connections, message processing, and client state.
"""

import json
import uuid
import logging
import asyncio
import websockets
from typing import Dict, Any

from config import DEFAULT_CONVERSATION_TURNS, AVAILABLE_VOICES
from utils import safe_send_response, get_connection_info
from models import model_manager
from audio_processor import audio_processor
from transcription_manager import transcription_manager
from prompt_logger import prompt_logger


class WebSocketHandler:
    """Handles WebSocket connections and message processing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_connections: Dict[str, Dict[str, Any]] = {}
    
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
                "session_created": False  # Track if session has been created
            }
            
            self.logger.info(f"✅ Client {client_address} connected with ID: {connection_id}")
            self.logger.info(f"📊 Active connections: {len(self.active_connections)}")
            
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
        """Handle binary audio message."""
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
        
        # Ensure session is created for this connection (only on first message)
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
            # Direct audio processing (legacy mode) - PARALLEL PROCESSING
            # -- Critical Path: Direct audio-to-AI response (fastest possible) --
            self.logger.info(f"Processing audio for AI response for {client_address}")
            
            # Get conversation context from transcription manager
            conversation_context = transcription_manager.get_conversation_context_for_ai(connection_id)
            
            # Combine with default system prompt
            combined_turns = connection["conversation_turns"].copy()
            combined_turns.extend(conversation_context)
            
            response_text = await model_manager.process_audio(audio_bytes, combined_turns, connection_id)
            self.logger.info(f"AI Response: {response_text[:100]}...")
            
            # Generate TTS audio from the response and send directly
            self.logger.info(f"Generating TTS audio for {client_address}")
            tts_audio = await audio_processor.generate_tts_audio(response_text)
            
            if tts_audio:
                # Send audio directly to vicidial bridge (no chunking needed)
                self.logger.info(f"Sending TTS audio directly to vicidial bridge: {len(tts_audio)} bytes")
                await websocket.send(tts_audio)
                self.logger.info(f"Successfully sent TTS audio to {client_address}")
            else:
                # Fallback: send text response if TTS fails
                self.logger.warning(f"TTS failed, sending text response to {client_address}")
                await safe_send_response(websocket, response_text, client_address)
            
            # Add AI response to history immediately after sending
            transcription_manager.add_ai_transcription(connection_id, response_text)
            
            # -- Background Task: Transcribe and log user input (non-blocking) --
            async def log_transcription(audio, conn_id):
                try:
                    user_transcription = await model_manager.transcribe_audio(audio, conn_id)
                    transcription_manager.add_user_transcription(conn_id, user_transcription)
                    
                    # Check if user provided their name and update session state
                    self._detect_and_update_candidate_name(conn_id, user_transcription)
                    
                    self.logger.info(f"[BACKGROUND] User transcription added for history: {user_transcription[:50]}...")
                except Exception as e:
                    self.logger.error(f"[BACKGROUND] Error transcribing audio for logging: {e}")
            
            # Fire off transcription task in background (no await - continues immediately)
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
        """Process transcribe request with parallel processing."""
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
        
        # -- Critical Path: Transcribe user input first, then generate AI response --
        self.logger.info(f"Processing audio for AI response for {client_address}")
        
        # STEP 1: Transcribe user input first (this is critical for proper context)
        self.logger.info(f"Transcribing user input for {client_address}")
        user_transcription = await model_manager.transcribe_audio(audio_bytes, connection_id)
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
        """Process features request."""
        self.logger.info(f"Processing features request for {client_address}")
        features_prompt = "Transcribe the audio and also provide speaker gender, emotion, accent, and audio quality."
        turns = [{"role": "system", "content": features_prompt}]
        response_text = await model_manager.process_audio(audio_bytes, turns)
        self.logger.info(f"Sending features response to {client_address}")
        await safe_send_response(websocket, response_text, client_address)
    
    async def _process_tts_request(self, websocket, client_address: str, audio_bytes: bytes, connection: Dict[str, Any]):
        """Process TTS request with parallel processing - sends audio directly to vicidial bridge."""
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
        
        # -- Critical Path: Transcribe user input first, then generate AI response --
        self.logger.info(f"Processing audio for AI response for {client_address}")
        
        # STEP 1: Transcribe user input first (this is critical for proper context)
        self.logger.info(f"Transcribing user input for {client_address}")
        user_transcription = await model_manager.transcribe_audio(audio_bytes, connection_id)
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
        
        # STEP 4: Add AI response to history
        transcription_manager.add_ai_transcription(connection_id, response_text)
        
        # STEP 5: Generate TTS audio from the response
        self.logger.info(f"Generating TTS audio for {client_address}")
        tts_audio = await audio_processor.generate_tts_audio(response_text)
        
        if tts_audio:
            # Send audio directly to vicidial bridge (no chunking needed)
            self.logger.info(f"Sending TTS audio directly to vicidial bridge: {len(tts_audio)} bytes")
            await websocket.send(tts_audio)
            self.logger.info(f"Successfully sent TTS audio to {client_address}")
        else:
            # Fallback: send text response if TTS fails
            self.logger.warning(f"TTS failed, sending text response to {client_address}")
            await safe_send_response(websocket, response_text, client_address)


# Global WebSocket handler instance
websocket_handler = WebSocketHandler()

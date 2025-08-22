from flask import Flask, render_template, send_from_directory, session, request
from flask_socketio import SocketIO, emit, join_room, leave_room
from src.utils.logger import setup_logger
from config.settings import WEB_SETTINGS
from src.web.routes import api
import os
import uuid
import json
import time

def create_app():
    """Create and configure the Flask application."""
    logger = setup_logger()
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SECRET_KEY"] = WEB_SETTINGS["secret_key"]
    app.config["DEBUG"] = WEB_SETTINGS["debug"]

    # Initialize SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

    # Register blueprints
    app.register_blueprint(api, url_prefix="/api")

    @app.route("/")
    def index():
        logger.info("Rendering web interface")
        # Generate session ID if not exists
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
            logger.info(f"Generated new session ID: {session['session_id']}")
        return render_template("index.html")
    
    @app.route("/test")
    def test_microphone():
        """Serve the microphone test page."""
        logger.info("Serving microphone test page")
        return send_from_directory(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "microphone_test.html")

    @app.route("/webrtc")
    def webrtc_interface():
        """Serve the WebRTC real-time voice interface."""
        logger.info("Serving WebRTC interface")
        return render_template("webrtc_interface.html")

    @app.route("/test-webrtc")
    def test_webrtc():
        """Serve the WebRTC connection test page."""
        logger.info("Serving WebRTC test page")
        return send_from_directory(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "test_webrtc_connection.html")

    @app.route("/test-sdp")
    def test_sdp():
        """Serve the SDP format test page."""
        logger.info("Serving SDP test page")
        return send_from_directory(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "test_sdp_format.html")

    # WebSocket event handlers
    @socketio.on('connect')
    def handle_connect():
        logger.info(f"Client connected: {request.sid}")
        emit('connected', {'data': 'Connected'})

    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info(f"Client disconnected: {request.sid}")

    @socketio.on('join')
    def handle_join(data):
        session_id = data.get('sessionId')
        join_room(session_id)
        logger.info(f"Client {request.sid} joined session {session_id}")
        emit('joined', {'sessionId': session_id, 'clientId': request.sid}, room=session_id)

    @socketio.on('offer')
    def handle_offer(data):
        session_id = data.get('sessionId')
        sdp = data.get('sdp')
        logger.info(f"Received offer for session {session_id} (not using WebRTC peer connection)")
        
        # For now, we're not using WebRTC peer connection
        # Just acknowledge the connection
        emit('answer', {'sdp': 'mock-sdp'}, room=session_id)

    @socketio.on('ice-candidate')
    def handle_ice_candidate(data):
        session_id = data.get('sessionId')
        candidate = data.get('candidate')
        emit('ice-candidate', {'candidate': candidate}, room=session_id, include_self=False)

    @socketio.on('audio-chunk')
    def handle_audio_chunk(data):
        session_id = data.get('sessionId')
        audio_data = data.get('audio')
        logger.info(f"Received audio chunk for session {session_id}, size: {len(audio_data) if audio_data else 0}")
        
        # Mock AI response
        mock_response = {
            'type': 'ai-response',
            'text': 'I received your audio message.',
            'audio_url': '/api/audio/mock_response.wav'
        }
        
        emit('ai-response', mock_response, room=session_id)

    def create_ai_answer(offer_sdp):
        """Create a mock SDP answer for the AI peer."""
        # For now, let's just echo back a minimal answer
        # In a real implementation, you'd parse the offer and create a proper answer
        logger.info(f"Creating answer for offer: {offer_sdp[:200]}...")
        
        # Create a minimal answer that should work with most offers
        answer_sdp = f"""v=0
o=- 1234567890 2 IN IP4 127.0.0.1
s=AI Voice Assistant
t=0 0
a=group:BUNDLE 0
m=audio 9 UDP/TLS/RTP/SAVPF 111
c=IN IP4 0.0.0.0
a=rtcp:9 IN IP4 0.0.0.0
a=ice-ufrag:ai123
a=ice-pwd:ai456
a=ice-options:trickle
a=fingerprint:sha-256 12:34:56:78:9A:BC:DE:F0:12:34:56:78:9A:BC:DE:F0:12:34:56:78:9A:BC:DE:F0:12:34:56:78:9A:BC:DE:F0
a=setup:passive
a=mid:0
a=sendrecv
a=rtcp-mux
a=rtpmap:111 opus/48000/2
a=fmtp:111 minptime=10;useinbandfec=1
a=ssrc:1234567890 cname:ai-voice
"""
        return answer_sdp

    logger.info("Flask app created successfully")
    return app, socketio

if __name__ == "__main__":
    app, socketio = create_app()
    
    logger = setup_logger()
    logger.info("Starting Flask app with SocketIO...")
    
    try:
        socketio.run(
            app,
            host=WEB_SETTINGS["host"],
            port=WEB_SETTINGS["port"],
            debug=WEB_SETTINGS["debug"],
            allow_unsafe_werkzeug=True
        )
    except KeyboardInterrupt:
        logger.info("Shutting down...")
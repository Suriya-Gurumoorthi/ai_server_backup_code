from flask import Blueprint, request, jsonify, send_file, session, render_template
import os
import time
from src.utils.logger import setup_logger
from src.stt.speech_recognizer import SpeechRecognizer
from src.tts.voice_generator import VoiceGenerator
from src.conversation.conversation_flow import ConversationFlow
from config.settings import STORAGE_SETTINGS
from config.roles import list_available_roles, get_role_names

api = Blueprint("api", __name__)
logger = setup_logger()

# Initialize components
stt = SpeechRecognizer()
tts = VoiceGenerator()

# Global conversation flows for different roles
conversation_flows = {}

def get_conversation_flow(role, session_id=None):
    """Get or create a conversation flow for the given role and session."""
    key = f"{role}_{session_id}" if session_id else role
    
    if key not in conversation_flows:
        conversation_flows[key] = ConversationFlow(role_name=role, session_id=session_id)
        logger.info(f"Created new conversation flow for key: {key}")
    
    return conversation_flows[key]

@api.route("/process_audio", methods=["POST"])
def process_audio():
    """Process uploaded audio file and return response."""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Get role from session or form data
        role = session.get('current_role', request.form.get('role', 'assistant'))
        session_id = session.get('session_id', 'default')
        logger.info(f"Processing audio for role: {role}, session: {session_id}")
        
        # Get file extension from original filename
        file_extension = os.path.splitext(audio_file.filename)[1] or '.wav'
        
        # Save uploaded file with original extension
        audio_path = os.path.join(STORAGE_SETTINGS['audio_dir'], f"upload_{id(audio_file)}{file_extension}")
        audio_file.save(audio_path)
        
        logger.info(f"Saved audio file: {audio_path}")
        logger.info(f"Audio file size: {os.path.getsize(audio_path)} bytes")
        
        # Convert speech to text
        logger.info("Starting speech recognition...")
        text = stt.recognize_speech(audio_path)
        if not text:
            logger.error("Speech recognition failed - no text returned")
            return jsonify({"error": "Could not recognize speech"}), 400
        
        logger.info(f"Speech recognition successful. Recognized text: '{text}'")
        
        # Generate response with role context
        logger.info(f"Generating AI response for text: '{text}' with role: {role}")
        conversation_with_role = get_conversation_flow(role, session_id)
        response = conversation_with_role.generate_response(text, [])
        logger.info(f"AI response generated: '{response}'")
        
        # Generate speech from response
        logger.info("Generating speech from AI response...")
        audio_output = tts.generate_speech(response)
        logger.info(f"Speech generation completed. Audio file: {audio_output}")
        
        result = {
            "text": text,
            "response": response,
            "audio_url": f"/api/audio/{os.path.basename(audio_output)}" if audio_output else None
        }
        
        logger.info(f"Final result: {result}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return jsonify({"error": str(e)}), 500

@api.route("/audio/<filename>")
def serve_audio(filename):
    """Serve generated audio files."""
    try:
        audio_path = os.path.join(STORAGE_SETTINGS['audio_dir'], filename)
        if os.path.exists(audio_path):
            return send_file(audio_path, mimetype='audio/wav')
        else:
            return jsonify({"error": "Audio file not found"}), 404
    except Exception as e:
        logger.error(f"Error serving audio: {e}")
        return jsonify({"error": str(e)}), 500

@api.route("/status")
def get_status():
    """Get system status."""
    return jsonify({
        "status": "running",
        "stt_available": stt.stt is not None,
        "tts_available": tts.tts.engine is not None
    })

@api.route("/test_audio", methods=["POST"])
def test_audio():
    """Test endpoint to verify audio upload is working."""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Get file extension from original filename
        file_extension = os.path.splitext(audio_file.filename)[1] or '.wav'
        
        # Save uploaded file with original extension
        audio_path = os.path.join(STORAGE_SETTINGS['audio_dir'], f"test_{id(audio_file)}{file_extension}")
        audio_file.save(audio_path)
        
        file_size = os.path.getsize(audio_path)
        
        logger.info(f"Test audio file saved: {audio_path}, size: {file_size} bytes")
        
        return jsonify({
            "success": True,
            "message": f"Audio file received successfully. Size: {file_size} bytes",
            "file_path": audio_path
        })
        
    except Exception as e:
        logger.error(f"Error in test_audio: {e}")
        return jsonify({"error": str(e)}), 500

@api.route("/process_realtime_audio", methods=["POST"])
def process_realtime_audio():
    """Process real-time audio chunks from WebRTC."""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        session_id = request.form.get('sessionId', 'unknown')
        
        if audio_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save real-time audio chunk
        audio_path = os.path.join(STORAGE_SETTINGS['audio_dir'], f"realtime_{session_id}_{int(time.time())}.wav")
        audio_file.save(audio_path)
        
        logger.info(f"Processing real-time audio for session {session_id}: {audio_path}")
        
        # Convert speech to text
        text = stt.recognize_speech(audio_path)
        if not text:
            logger.warning(f"No speech detected in real-time audio for session {session_id}")
            return jsonify({"error": "No speech detected"}), 400
        
        logger.info(f"Real-time STT result for session {session_id}: '{text}'")
        
        # Generate AI response
        role = session.get('current_role', 'assistant')
        conversation_with_role = get_conversation_flow(role, session_id)
        response = conversation_with_role.generate_response(text, [])
        
        logger.info(f"AI response for session {session_id}: '{response}'")
        
        # Generate speech from response
        audio_output = tts.generate_speech(response)
        
        result = {
            "text": text,
            "response": response,
            "audio_url": f"/api/audio/{os.path.basename(audio_output)}" if audio_output else None,
            "session_id": session_id
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing real-time audio: {e}")
        return jsonify({"error": str(e)}), 500

@api.route("/process_text", methods=["POST"])
def process_text():
    """Process text input and return response."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text']
        role = session.get('current_role', data.get('role', 'assistant'))
        session_id = session.get('session_id', 'default')
        
        if not text.strip():
            return jsonify({"error": "Empty text provided"}), 400
        
        # Generate response with role context
        conversation_with_role = get_conversation_flow(role, session_id)
        response = conversation_with_role.generate_response(text, [])
        
        # Generate speech from response
        audio_output = tts.generate_speech(response)
        
        return jsonify({
            "text": text,
            "response": response,
            "audio_url": f"/api/audio/{os.path.basename(audio_output)}" if audio_output else None
        })
        
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        return jsonify({"error": str(e)}), 500

@api.route("/clear_memory", methods=["POST"])
def clear_memory():
    """Clear conversation memory for a new session."""
    try:
        data = request.get_json()
        role = data.get('role', 'assistant') if data else 'assistant'
        session_id = session.get('session_id', 'default')
        
        logger.info(f"Clearing conversation memory for role: {role}, session: {session_id}")
        
        # Create a new conversation flow instance to clear memory
        conversation_with_role = get_conversation_flow(role, session_id)
        conversation_with_role.clear_memory()
        
        return jsonify({
            "success": True,
            "message": f"Conversation memory cleared for {role}",
            "memory_summary": conversation_with_role.get_memory_summary()
        })
        
    except Exception as e:
        logger.error(f"Error clearing memory: {e}")
        return jsonify({"error": str(e)}), 500

@api.route("/memory_summary", methods=["GET"])
def get_memory_summary():
    """Get a summary of the current conversation memory."""
    try:
        role = request.args.get('role', 'assistant')
        session_id = session.get('session_id', 'default')
        conversation_with_role = get_conversation_flow(role, session_id)
        
        return jsonify({
            "success": True,
            "memory_summary": conversation_with_role.get_memory_summary()
        })
        
    except Exception as e:
        logger.error(f"Error getting memory summary: {e}")
        return jsonify({"error": str(e)}), 500

@api.route("/test_memory", methods=["POST"])
def test_memory():
    """Test endpoint to verify conversation memory is working."""
    try:
        data = request.get_json()
        text = data.get('text', 'Hello')
        role = data.get('role', 'assistant')
        session_id = session.get('session_id', 'default')
        
        logger.info(f"Testing memory with text: '{text}' for role: {role}, session: {session_id}")
        
        # Create conversation flow and test memory
        conversation_with_role = get_conversation_flow(role, session_id)
        
        # Get initial memory state
        initial_memory = conversation_with_role.get_memory_summary()
        
        # Generate first response
        response1 = conversation_with_role.generate_response(text, [])
        
        # Get memory after first response
        memory_after_1 = conversation_with_role.get_memory_summary()
        
        # Generate second response to test context
        response2 = conversation_with_role.generate_response("What did I just ask you?", [])
        
        # Get final memory state
        final_memory = conversation_with_role.get_memory_summary()
        
        return jsonify({
            "success": True,
            "initial_memory": initial_memory,
            "memory_after_1": memory_after_1,
            "final_memory": final_memory,
            "first_response": response1,
            "second_response": response2,
            "context_working": "What did I just ask you?" in response2 or text.lower() in response2.lower()
        })
        
    except Exception as e:
        logger.error(f"Error testing memory: {e}")
        return jsonify({"error": str(e)}), 500

@api.route("/roles", methods=["GET"])
def list_roles():
    """List all available roles."""
    try:
        roles = get_role_names()
        return jsonify({
            "success": True,
            "roles": roles,
            "current_role": session.get('current_role', 'assistant')
        })
    except Exception as e:
        logger.error(f"Error listing roles: {e}")
        return jsonify({"error": str(e)}), 500

@api.route("/change_role", methods=["POST"])
def change_role():
    """Change the current role for the session."""
    try:
        data = request.get_json()
        new_role = data.get('role', 'assistant')
        session_id = session.get('session_id', 'default')
        
        logger.info(f"Changing role to: {new_role} for session: {session_id}")
        
        # Clear the old conversation flow from cache
        old_role = session.get('current_role', 'assistant')
        old_key = f"{old_role}_{session_id}"
        if old_key in conversation_flows:
            del conversation_flows[old_key]
            logger.info(f"Removed old conversation flow: {old_key}")
        
        # Get or create conversation flow with new role
        conversation_flow = get_conversation_flow(new_role, session_id)
        
        # Clear memory for the new role to ensure clean transition
        conversation_flow.clear_memory()
        
        # Update session
        session['current_role'] = new_role
        
        return jsonify({
            "success": True,
            "message": f"Role changed to {new_role}",
            "current_role": new_role,
            "memory_summary": conversation_flow.get_memory_summary()
        })
        
    except Exception as e:
        logger.error(f"Error changing role: {e}")
        return jsonify({"error": str(e)}), 500

@api.route("/current_role", methods=["GET"])
def get_current_role():
    """Get the current role for the session."""
    try:
        current_role = session.get('current_role', 'assistant')
        session_id = session.get('session_id', 'default')
        
        # Get conversation flow to check actual role
        conversation_flow = get_conversation_flow(current_role, session_id)
        actual_role = conversation_flow.get_current_role()
        
        return jsonify({
            "success": True,
            "current_role": actual_role,
            "session_role": current_role
        })
        
    except Exception as e:
        logger.error(f"Error getting current role: {e}")
        return jsonify({"error": str(e)}), 500

@api.route("/webrtc")
def webrtc_interface():
    """Serve the WebRTC real-time voice interface."""
    return render_template('webrtc_interface.html')
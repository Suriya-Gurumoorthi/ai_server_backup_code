#!/usr/bin/env python3
"""
Persistent Model Server for Ultravox - Keeps model loaded across script runs
"""

import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import urllib.parse
from model_loader import get_ultravox_pipeline, is_model_loaded
from ultravox_usage import chat_with_audio, answer_question, transcribe_audio, creative_response

class ModelRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        """Handle POST requests for model inference"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            # Parse JSON request
            request = json.loads(post_data.decode('utf-8'))
            action = request.get('action', 'chat')
            
            # Get model pipeline
            pipeline = get_ultravox_pipeline()
            if pipeline is None:
                self.send_error_response("Model not loaded")
                return
            
            # Handle different actions
            if action == 'chat':
                result = chat_with_audio(
                    audio_file_path=request.get('audio_file'),
                    user_message=request.get('message', 'Hello'),
                    system_prompt=request.get('system_prompt'),
                    max_tokens=request.get('max_tokens', 50)
                )
            elif action == 'transcribe':
                result = transcribe_audio(
                    audio_file_path=request.get('audio_file'),
                    system_prompt=request.get('system_prompt')
                )
            elif action == 'answer':
                result = answer_question(
                    question=request.get('question'),
                    audio_file_path=request.get('audio_file'),
                    system_prompt=request.get('system_prompt')
                )
            elif action == 'creative':
                result = creative_response(
                    prompt=request.get('prompt'),
                    audio_file_path=request.get('audio_file'),
                    system_prompt=request.get('system_prompt')
                )
            else:
                self.send_error_response(f"Unknown action: {action}")
                return
            
            # Send response
            self.send_success_response(result)
            
        except Exception as e:
            self.send_error_response(str(e))
    
    def do_GET(self):
        """Handle GET requests for status"""
        if self.path == '/status':
            status = {
                'model_loaded': is_model_loaded(),
                'timestamp': time.time()
            }
            self.send_success_response(status)
        else:
            self.send_error_response("Invalid endpoint")
    
    def send_success_response(self, data):
        """Send successful response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def send_error_response(self, error_message):
        """Send error response"""
        self.send_response(400)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({'error': error_message}).encode())
    
    def log_message(self, format, *args):
        """Suppress HTTP server logs"""
        pass

def start_server(port=8000):
    """Start the model server"""
    server = HTTPServer(('localhost', port), ModelRequestHandler)
    print(f"🚀 Model server started on http://localhost:{port}")
    print("📦 Loading model...")
    
    # Load model in background
    def load_model():
        get_ultravox_pipeline()
        print("✅ Model loaded and ready!")
    
    threading.Thread(target=load_model, daemon=True).start()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        server.shutdown()

if __name__ == "__main__":
    start_server()



#!/usr/bin/env python3
"""
Comprehensive Ultravox API Server
Implements all major Ultravox endpoints for multimodal AI interactions
"""

import json
import base64
import io
import logging
import os
import tempfile
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import librosa
import numpy as np
from transformers import pipeline
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model pipeline - loaded once at startup
logger.info("Loading Ultravox model...")
pipe = pipeline(
    model="fixie-ai/ultravox-v0_5-llama-3_2-1b", 
    trust_remote_code=True
)
logger.info("Model loaded successfully!")

class UltravoxAPIHandler(BaseHTTPRequestHandler):
    """Comprehensive Ultravox API handler supporting all major endpoints"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _send_response(self, status_code=200, data=None, content_type="application/json"):
        """Send JSON response with proper headers"""
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
        
        if data is not None:
            if isinstance(data, dict) or isinstance(data, list):
                response = json.dumps(data, indent=2)
            else:
                response = str(data)
            self.wfile.write(response.encode('utf-8'))
    
    def _send_error(self, status_code, message):
        """Send error response"""
        self._send_response(status_code, {"error": message})
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query_params = parse_qs(parsed_path.query)
        
        try:
            if path == '/':
                self._get_api_info()
            elif path == '/models':
                self._get_models()
            elif path == '/calls':
                self._get_calls(query_params)
            elif path == '/health':
                self._get_health()
            elif path == '/docs':
                self._get_docs()
            elif path == '/test-audio':
                self._get_test_audio()
            elif path.startswith('/calls/'):
                call_id = path.split('/')[-1]
                self._get_call(call_id)
            else:
                self._send_error(404, "Endpoint not found")
        except Exception as e:
            logger.error(f"Error in GET {path}: {e}")
            self._send_error(500, f"Internal server error: {str(e)}")
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        try:
            if path == '/generate':
                self._post_generate()
            elif path == '/chat/audio/binary':
                self._post_chat_audio_binary()
            elif path == '/chat/text':
                self._post_chat_text()
            elif path == '/chat/multimodal':
                self._post_chat_multimodal()
            elif path == '/calls':
                self._post_create_call()
            elif path.startswith('/calls/') and path.endswith('/stages'):
                call_id = path.split('/')[2]
                self._post_create_stage(call_id)
            else:
                self._send_error(404, "Endpoint not found")
        except Exception as e:
            logger.error(f"Error in POST {path}: {e}")
            self._send_error(500, f"Internal server error: {str(e)}")
    
    def do_PUT(self):
        """Handle PUT requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        try:
            if path.startswith('/calls/'):
                call_id = path.split('/')[-1]
                self._put_update_call(call_id)
            else:
                self._send_error(404, "Endpoint not found")
        except Exception as e:
            logger.error(f"Error in PUT {path}: {e}")
            self._send_error(500, f"Internal server error: {str(e)}")
    
    def do_DELETE(self):
        """Handle DELETE requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        try:
            if path.startswith('/calls/'):
                call_id = path.split('/')[-1]
                self._delete_call(call_id)
            else:
                self._send_error(404, "Endpoint not found")
        except Exception as e:
            logger.error(f"Error in DELETE {path}: {e}")
            self._send_error(500, f"Internal server error: {str(e)}")
    
    def _get_api_info(self):
        """Get API information and available endpoints"""
        info = {
            "name": "UltraVox API Server",
            "version": "1.0.0",
            "model": "fixie-ai/ultravox-v0_5-llama-3_2-1b",
            "description": "Comprehensive Ultravox API server supporting multimodal AI interactions",
            "endpoints": {
                "GET /": "API information",
                "GET /health": "Health check",
                "GET /models": "List available models",
                "GET /calls": "List calls",
                "GET /calls/{id}": "Get specific call",
                "POST /generate": "Generate text response",
                "POST /chat/audio/binary": "Process audio input",
                "POST /chat/text": "Process text input",
                "POST /calls": "Create new call",
                "POST /calls/{id}/stages": "Create call stage",
                "PUT /calls/{id}": "Update call",
                "DELETE /calls/{id}": "Delete call"
            }
        }
        self._send_response(200, info)
    
    def _get_health(self):
        """Health check endpoint"""
        health = {
        "status": "healthy",
            "model_loaded": pipe is not None,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "timestamp": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "CPU"
        }
        self._send_response(200, health)
    
    def _get_docs(self):
        """API documentation endpoint"""
        docs = {
            "title": "UltraVox API Documentation",
            "version": "1.0.0",
            "description": "Comprehensive API for Ultravox multimodal AI interactions",
            "endpoints": {
                "GET /": {
                    "description": "Get API information and available endpoints",
                    "response": "JSON object with API details"
                },
                "GET /health": {
                    "description": "Health check endpoint",
                    "response": "JSON object with server and model status"
                },
                "GET /docs": {
                    "description": "API documentation (this endpoint)",
                    "response": "JSON object with detailed API documentation"
                },
                "GET /models": {
                    "description": "List available models",
                    "response": "JSON array of available models"
                },
                "POST /generate": {
                    "description": "Generate text from a prompt",
                    "body": {
                        "prompt": "string (required)",
                        "max_length": "integer (optional, default: 100)",
                        "temperature": "float (optional, default: 0.7)"
                    },
                    "response": "JSON object with generated text"
                },
                "POST /chat/audio/binary": {
                    "description": "Process audio input and generate response",
                    "body": "Multipart form data with audio file (optional: prompt/text field)",
                    "example": "curl -X POST -F 'audio=@file.wav' http://localhost:8000/chat/audio/binary",
                    "example_with_prompt": "curl -X POST -F 'prompt=Describe this audio' -F 'audio=@file.wav' http://localhost:8000/chat/audio/binary",
                    "response": "JSON object with generated text from audio",
                    "note": "Uses correct Ultravox format: turns with text+role and audio array"
                },
                "POST /chat/multimodal": {
                    "description": "Process multimodal input (text + audio)",
                    "body": "Multipart form data with text and audio file",
                    "example": "curl -X POST -F 'text=Describe this audio' -F 'audio=@file.wav' http://localhost:8000/chat/multimodal",
                    "response": "JSON object with generated text from multimodal input"
                },
                "POST /chat/text": {
                    "description": "Chat with text input and conversation history",
                    "body": {
                        "message": "string (required)",
                        "history": "array of conversation turns (optional)"
                    },
                    "response": "JSON object with chat response"
                },
                "GET /calls": {
                    "description": "List all calls",
                    "response": "JSON array of calls"
                },
                "POST /calls": {
                    "description": "Create a new call",
                    "body": "JSON object with call configuration",
                    "response": "JSON object with created call details"
                }
            },
            "examples": {
                "text_generation": "curl -X POST http://localhost:8000/generate -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello world\"}'",
                "audio_processing": "curl -X POST http://localhost:8000/chat/audio/binary -F 'audio=@audio.wav'",
                "chat": "curl -X POST http://localhost:8000/chat/text -H 'Content-Type: application/json' -d '{\"message\": \"How are you?\"}'"
            }
        }
        self._send_response(200, docs)
    
    def _get_test_audio(self):
        """Test endpoint to help debug audio uploads"""
        test_info = {
            "title": "Audio Upload Test Endpoint",
            "description": "Use this endpoint to test audio file uploads",
            "supported_formats": ["WAV", "MP3", "FLAC"],
            "min_file_size": "44 bytes (WAV header)",
            "test_commands": {
                "curl_audio_only": "curl -X POST -F 'audio=@test.wav' http://localhost:8000/chat/audio/binary",
                "curl_audio_with_prompt": "curl -X POST -F 'prompt=Describe this audio' -F 'audio=@test.wav' http://localhost:8000/chat/audio/binary",
                "curl_multimodal": "curl -X POST -F 'text=Describe this audio' -F 'audio=@test.wav' http://localhost:8000/chat/multimodal",
                "curl_binary": "curl -X POST -H 'Content-Type: audio/wav' --data-binary @test.wav http://localhost:8000/chat/audio/binary"
            },
            "debugging": {
                "check_file_size": "Make sure your audio file is > 44 bytes",
                "check_format": "Ensure the file is a valid WAV/MP3 format",
                "check_curl": "Use -F flag for multipart uploads"
            }
        }
        self._send_response(200, test_info)
    
    def _get_models(self):
        """List available models"""
        models = {
            "models": [
                {
                    "id": "fixie-ai/ultravox-v0_5-llama-3_2-1b",
                    "name": "UltraVox v0.5 Llama 3.2 1B",
                    "type": "multimodal",
                    "capabilities": ["text", "audio"],
                    "description": "Multimodal AI model supporting text and audio interactions"
                }
            ]
        }
        self._send_response(200, models)
    
    def _get_calls(self, query_params):
        """List calls with optional filtering"""
        # In a real implementation, this would query a database
        calls = {
            "calls": [
                {
                    "id": "call_123",
                    "status": "active",
                    "created_at": "2024-01-01T00:00:00Z",
                    "model": "fixie-ai/ultravox-v0_5-llama-3_2-1b"
                }
            ],
            "total": 1
        }
        self._send_response(200, calls)
    
    def _get_call(self, call_id):
        """Get specific call details"""
        call = {
            "id": call_id,
            "status": "active",
            "created_at": "2024-01-01T00:00:00Z",
            "model": "fixie-ai/ultravox-v0_5-llama-3_2-1b",
            "stages": []
        }
        self._send_response(200, call)
    
    def _post_generate(self):
        """Generate text response from prompt"""
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            prompt = data.get("prompt", "")
            max_length = data.get("max_length", 100)
            temperature = data.get("temperature", 0.7)
            
            if not prompt:
                self._send_error(400, "Prompt is required")
                return
            
            # Prepare input for Ultravox model
            inputs = {"turns": [{"text": prompt, "role": "user"}]}
            
            # Generate response
            result = pipe(inputs)
            
            response = {
                "prompt": prompt,
                "generated_text": result,
                "max_length": max_length,
                "temperature": temperature
            }
            
            self._send_response(200, response)
            
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON in request body")
        except Exception as e:
            logger.error(f"Error in text generation: {e}")
            self._send_error(500, f"Generation error: {str(e)}")
    
    def _post_chat_audio_binary(self):
        """Process audio input and generate response"""
        try:
            # Check if this is a multipart form upload
            content_type = self.headers.get('Content-Type', '')
            logger.info(f"Content-Type: {content_type}")
            
            if 'multipart/form-data' in content_type:
                # Handle multipart form data (curl -F)
                logger.info("Handling multipart form data")
                self._handle_multipart_audio()
            else:
                # Handle raw binary audio data
                logger.info("Handling binary audio data")
                self._handle_binary_audio()
                
        except Exception as e:
            logger.error(f"Error in audio processing: {e}", exc_info=True)
            self._send_error(500, f"Audio processing error: {str(e)}")
    
    def _handle_multipart_audio(self):
        """Handle multipart form data audio upload"""
        import cgi
        import io
        
        try:
            # Parse multipart form data
            ctype, pdict = cgi.parse_header(self.headers['Content-Type'])
            logger.info(f"Parsed content type: {ctype}, params: {pdict}")
            
            if 'boundary' not in pdict:
                logger.error("No boundary found in Content-Type header")
                self._send_error(400, "Invalid multipart form data: no boundary")
                return
            
            pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
            
            # Read the form data
            content_length = int(self.headers.get('Content-Length', 0))
            logger.info(f"Content length: {content_length}")
            
            if content_length == 0:
                logger.error("Empty content length")
                self._send_error(400, "Empty request body")
                return
            
            post_data = self.rfile.read(content_length)
            logger.info(f"Read {len(post_data)} bytes of data")
            
            if len(post_data) != content_length:
                logger.error(f"Content length mismatch: expected {content_length}, got {len(post_data)}")
                self._send_error(400, "Content length mismatch")
                return
            
            # Parse the multipart data
            form = cgi.parse_multipart(io.BytesIO(post_data), pdict)
            logger.info(f"Form keys: {list(form.keys())}")
            
            # Get optional text prompt
            text_prompt = ""
            if 'prompt' in form and form['prompt']:
                text_prompt = form['prompt'][0].decode('utf-8')
                logger.info(f"Text prompt: {text_prompt}")
            elif 'text' in form and form['text']:
                text_prompt = form['text'][0].decode('utf-8')
                logger.info(f"Text prompt: {text_prompt}")
            
            # Get the audio file - try different possible field names
            audio_data = None
            possible_field_names = ['audio', 'file', 'upload', 'data']
            
            for field_name in possible_field_names:
                if field_name in form and form[field_name]:
                    audio_data = form[field_name][0]
                    logger.info(f"Found audio data in field '{field_name}': {len(audio_data)} bytes")
                    break
            
            if audio_data is None:
                logger.error(f"No audio file found in form data. Available fields: {list(form.keys())}")
                self._send_error(400, f"No audio file found in form data. Available fields: {list(form.keys())}")
                return
            
            if len(audio_data) == 0:
                logger.error("Audio data is empty")
                self._send_error(400, "Audio file is empty")
                return
            
            logger.info(f"Audio data size: {len(audio_data)} bytes")
            
        except Exception as e:
            logger.error(f"Error parsing multipart data: {e}", exc_info=True)
            self._send_error(400, f"Error parsing form data: {str(e)}")
            return
        
        # Save audio data to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        # Validate audio file size
        file_size = os.path.getsize(temp_file_path)
        logger.info(f"Audio file size: {file_size} bytes")
        
        if file_size < 44:  # WAV header is at least 44 bytes
            logger.error(f"Audio file too small: {file_size} bytes (minimum 44 for WAV)")
            self._send_error(400, f"Audio file too small: {file_size} bytes")
            os.unlink(temp_file_path)
            return
        
        try:
            # Try SoundFile first (better for WAV files)
            try:
                import soundfile as sf
                logger.info("Trying SoundFile to read audio...")
                audio, sr = sf.read(temp_file_path)
                logger.info(f"Audio loaded with SoundFile: shape={audio.shape}, sr={sr}")
            except ImportError:
                logger.info("SoundFile not available, falling back to librosa...")
                # Fallback to librosa
                audio, sr = librosa.load(temp_file_path, sr=16000)
                logger.info(f"Audio loaded with librosa: shape={audio.shape}, sr={sr}")
            except Exception as e:
                logger.error(f"SoundFile failed: {e}, trying librosa...")
                # Fallback to librosa
                audio, sr = librosa.load(temp_file_path, sr=16000)
                logger.info(f"Audio loaded with librosa: shape={audio.shape}, sr={sr}")
            
            # Prepare input for Ultravox model with audio
            # Ultravox requires a text turn with "role" field before audio turn
            if text_prompt:
                # Use provided text prompt
                inputs = {
                    "turns": [
                        {"text": text_prompt, "role": "user"},
                        {"audio": audio}
                    ],
                    "sampling_rate": sr
                }
                logger.info(f"Prepared inputs for model: turns with prompt '{text_prompt}' and audio array of shape {audio.shape}")
            else:
                # Use empty text for audio-only processing
                inputs = {
                    "turns": [
                        {"text": "", "role": "user"},  # Empty text with role for audio-only processing
                        {"audio": audio}
                    ],
                    "sampling_rate": sr
                }
                logger.info(f"Prepared inputs for model: turns with empty text+role and audio array of shape {audio.shape}")
            
            # Generate response
            logger.info("Generating response from model...")
            result = pipe(inputs)
            logger.info(f"Model response: {result}")
            
            response = {
                "input_type": "audio",
                "text_prompt": text_prompt if text_prompt else "",
                "audio_length": len(audio) / sr,
                "sampling_rate": sr,
                "generated_text": result
            }
            
            self._send_response(200, response)
            
        except Exception as e:
            logger.error(f"Error processing audio in multipart handler: {e}", exc_info=True)
            self._send_error(500, f"Audio processing error: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def _handle_binary_audio(self):
        """Handle raw binary audio data"""
        content_length = int(self.headers.get('Content-Length', 0))
        audio_data = self.rfile.read(content_length)
        
        # Save audio data to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        try:
            # Load and process audio
            audio, sr = librosa.load(temp_file_path, sr=16000)
            
            # Prepare input for Ultravox model with audio
            # Ultravox requires a text turn with "role" field before audio turn
            inputs = {
                "turns": [
                    {"text": "", "role": "user"},  # Empty text with role for audio-only processing
                    {"audio": audio}
                ],
                "sampling_rate": sr
            }
            logger.info(f"Prepared inputs for model: turns with text+role and audio array of shape {audio.shape}")
            
            # Generate response
            logger.info("Generating response from model...")
            result = pipe(inputs)
            logger.info(f"Model response: {result}")
            
            response = {
                "input_type": "audio",
                "audio_length": len(audio) / sr,
                "sampling_rate": sr,
                "generated_text": result
            }
            
            self._send_response(200, response)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def _post_chat_text(self):
        """Process text input for chat"""
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            message = data.get("message", "")
            conversation_history = data.get("history", [])
            
            if not message:
                self._send_error(400, "Message is required")
                return
            
            # Prepare conversation turns
            turns = []
            for turn in conversation_history:
                turns.append(turn)
            turns.append({"text": message, "role": "user"})
            
            inputs = {"turns": turns}
            
            # Generate response
            result = pipe(inputs)
            
            response = {
                "message": message,
                "response": result,
                "conversation_length": len(turns)
            }
            
            self._send_response(200, response)
            
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON in request body")
        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            self._send_error(500, f"Chat processing error: {str(e)}")
    
    def _post_chat_multimodal(self):
        """Process multimodal input (text + audio)"""
        try:
            # Check if this is a multipart form upload
            content_type = self.headers.get('Content-Type', '')
            logger.info(f"Multimodal Content-Type: {content_type}")
            
            if 'multipart/form-data' in content_type:
                # Handle multipart form data with text and audio
                self._handle_multimodal_multipart()
            else:
                self._send_error(400, "Multimodal endpoint requires multipart form data")
                
        except Exception as e:
            logger.error(f"Error in multimodal processing: {e}", exc_info=True)
            self._send_error(500, f"Multimodal processing error: {str(e)}")
    
    def _handle_multimodal_multipart(self):
        """Handle multimodal multipart form data (text + audio)"""
        import cgi
        import io
        
        try:
            # Parse multipart form data
            ctype, pdict = cgi.parse_header(self.headers['Content-Type'])
            logger.info(f"Multimodal parsed content type: {ctype}, params: {pdict}")
            
            if 'boundary' not in pdict:
                logger.error("No boundary found in Content-Type header")
                self._send_error(400, "Invalid multipart form data: no boundary")
                return
            
            pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
            
            # Read the form data
            content_length = int(self.headers.get('Content-Length', 0))
            logger.info(f"Multimodal content length: {content_length}")
            
            if content_length == 0:
                logger.error("Empty content length")
                self._send_error(400, "Empty request body")
                return
            
            post_data = self.rfile.read(content_length)
            logger.info(f"Read {len(post_data)} bytes of multimodal data")
            
            # Parse the multipart data
            form = cgi.parse_multipart(io.BytesIO(post_data), pdict)
            logger.info(f"Multimodal form keys: {list(form.keys())}")
            
            # Get text and audio
            text_prompt = ""
            if 'text' in form and form['text']:
                text_prompt = form['text'][0].decode('utf-8')
                logger.info(f"Text prompt: {text_prompt}")
            
            # Get audio file
            audio_data = None
            possible_field_names = ['audio', 'file', 'upload', 'data']
            
            for field_name in possible_field_names:
                if field_name in form and form[field_name]:
                    audio_data = form[field_name][0]
                    logger.info(f"Found audio data in field '{field_name}': {len(audio_data)} bytes")
                    break
            
            if audio_data is None:
                logger.error("No audio file found in multimodal form data")
                self._send_error(400, "No audio file found in form data")
                return
            
            if len(audio_data) == 0:
                logger.error("Audio data is empty")
                self._send_error(400, "Audio file is empty")
                return
            
            # Save and process audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                # Load audio
                try:
                    import soundfile as sf
                    audio, sr = sf.read(temp_file_path)
                    logger.info(f"Audio loaded with SoundFile: shape={audio.shape}, sr={sr}")
                except ImportError:
                    audio, sr = librosa.load(temp_file_path, sr=16000)
                    logger.info(f"Audio loaded with librosa: shape={audio.shape}, sr={sr}")
                except Exception as e:
                    logger.error(f"SoundFile failed: {e}, trying librosa...")
                    audio, sr = librosa.load(temp_file_path, sr=16000)
                    logger.info(f"Audio loaded with librosa: shape={audio.shape}, sr={sr}")
                
                # Prepare multimodal input
                turns = []
                if text_prompt:
                    turns.append({"text": text_prompt, "role": "user"})
                turns.append({"audio": audio})
                
                inputs = {
                    "turns": turns,
                    "sampling_rate": sr
                }
                logger.info(f"Prepared multimodal inputs: {len(turns)} turns")
                
                # Generate response
                logger.info("Generating multimodal response from model...")
                result = pipe(inputs)
                logger.info(f"Multimodal model response: {result}")
                
                response = {
                    "input_type": "multimodal",
                    "text_prompt": text_prompt,
                    "audio_length": len(audio) / sr,
                    "sampling_rate": sr,
                    "generated_text": result
                }
                
                self._send_response(200, response)
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            
        except Exception as e:
            logger.error(f"Error parsing multimodal data: {e}", exc_info=True)
            self._send_error(400, f"Error parsing multimodal form data: {str(e)}")
    
    def _post_create_call(self):
        """Create a new call"""
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            
            # Generate a new call ID
            call_id = f"call_{hash(str(data)) % 1000000}"
            
            call = {
                "id": call_id,
                "status": "created",
                "created_at": "2024-01-01T00:00:00Z",
                "model": "fixie-ai/ultravox-v0_5-llama-3_2-1b",
                "config": data
            }
            
            self._send_response(201, call)
            
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON in request body")
        except Exception as e:
            logger.error(f"Error creating call: {e}")
            self._send_error(500, f"Call creation error: {str(e)}")
    
    def _post_create_stage(self, call_id):
        """Create a new stage for a call"""
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            
            stage = {
                "id": f"stage_{hash(str(data)) % 1000000}",
                "call_id": call_id,
                "type": data.get("type", "text"),
                "status": "created",
                "created_at": "2024-01-01T00:00:00Z"
            }
            
            self._send_response(201, stage)
            
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON in request body")
        except Exception as e:
            logger.error(f"Error creating stage: {e}")
            self._send_error(500, f"Stage creation error: {str(e)}")
    
    def _put_update_call(self, call_id):
        """Update an existing call"""
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            
            call = {
                "id": call_id,
                "status": "updated",
                "updated_at": "2024-01-01T00:00:00Z",
                "config": data
            }
            
            self._send_response(200, call)
            
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON in request body")
        except Exception as e:
            logger.error(f"Error updating call: {e}")
            self._send_error(500, f"Call update error: {str(e)}")
    
    def _delete_call(self, call_id):
        """Delete a call"""
        response = {
            "id": call_id,
            "status": "deleted",
            "deleted_at": "2024-01-01T00:00:00Z"
        }
        
        self._send_response(200, response)
    
    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info(f"{self.address_string()} - {format % args}")

def main():
    """Main function to start the server"""
    logger.info("Starting UltraVox API Server...")
    
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, UltravoxAPIHandler)
    
    logger.info("Server running on http://localhost:8000")
    logger.info("Available endpoints:")
    logger.info("  GET  / - API information")
    logger.info("  GET  /health - Health check")
    logger.info("  GET  /models - List models")
    logger.info("  GET  /calls - List calls")
    logger.info("  POST /generate - Generate text")
    logger.info("  POST /chat/audio/binary - Process audio")
    logger.info("  POST /chat/text - Chat with text")
    logger.info("  POST /calls - Create call")
    logger.info("Press Ctrl+C to stop the server")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        httpd.server_close()

if __name__ == "__main__":
    main()
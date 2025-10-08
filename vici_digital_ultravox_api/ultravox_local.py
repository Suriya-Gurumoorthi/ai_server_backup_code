import asyncio
import base64
import json
import logging
import os
import socket
import struct
import threading
import numpy as np
from array import array
import websockets
from scipy import signal
import hashlib
import hmac
import time
import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForCausalLM
from ultravox import Ultravox

# Configuration
LISTEN_HOST, LISTEN_PORT = "0.0.0.0", 9092

# Security configuration
SECRET_KEY = os.getenv("BRIDGE_SECRET_KEY", "your-secret-key-change-this")
ALLOWED_IPS = os.getenv("ALLOWED_IPS", "").split(",")  # Comma-separated IPs
ENABLE_AUTH = os.getenv("ENABLE_AUTH", "true").lower() == "true"

# Ultravox model configuration
ULTRAVOX_MODEL_PATH = os.getenv("ULTRAVOX_MODEL_PATH", "fixie-ai/ultravox")
ULTRAVOX_DEVICE = os.getenv("ULTRAVOX_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Audio processing constants
VICIDIAL_SAMPLE_RATE = 8000
ULTRAVOX_SAMPLE_RATE = 48000
UPSAMPLE_FACTOR = ULTRAVOX_SAMPLE_RATE // VICIDIAL_SAMPLE_RATE  # 6
BYTES_PER_FRAME = 320  # 160 samples * 2 bytes per sample for 16-bit audio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityManager:
    """Handles authentication and authorization for remote connections"""
    
    def __init__(self):
        self.secret_key = SECRET_KEY.encode('utf-8')
        self.allowed_ips = [ip.strip() for ip in ALLOWED_IPS if ip.strip()]
    
    def verify_ip(self, client_ip):
        """Verify if client IP is allowed"""
        if not self.allowed_ips:  # If no IPs specified, allow all
            return True
        return client_ip in self.allowed_ips
    
    def generate_token(self, timestamp):
        """Generate authentication token"""
        message = f"ultravox_bridge:{timestamp}".encode('utf-8')
        signature = hmac.new(self.secret_key, message, hashlib.sha256).hexdigest()
        return f"{timestamp}:{signature}"
    
    def verify_token(self, token):
        """Verify authentication token"""
        try:
            timestamp_str, signature = token.split(':', 1)
            timestamp = int(timestamp_str)
            
            # Check if token is not too old (5 minutes)
            if time.time() - timestamp > 300:
                return False
            
            expected_signature = hmac.new(
                self.secret_key, 
                f"ultravox_bridge:{timestamp}".encode('utf-8'), 
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
        except:
            return False

class AudioProcessor:
    """Handles audio upsampling and downsampling between Vicidial and Ultravox"""
    
    def __init__(self):
        # Design low-pass filter for downsampling (anti-aliasing filter)
        # Cutoff frequency should be less than half the target sample rate
        cutoff_freq = VICIDIAL_SAMPLE_RATE / 2.2  # Slightly below Nyquist
        self.downsample_filter = signal.butter(4, cutoff_freq, fs=ULTRAVOX_SAMPLE_RATE, btype='low')[1]
        
        # Design low-pass filter for upsampling (interpolation filter)
        cutoff_freq = VICIDIAL_SAMPLE_RATE / 2.2
        self.upsample_filter = signal.butter(4, cutoff_freq, fs=ULTRAVOX_SAMPLE_RATE, btype='low')[1]
    
    def upsample_8k_to_48k(self, audio_data):
        """
        Upsample 8kHz audio to 48kHz for Ultravox
        audio_data: bytes or numpy array of 8kHz audio
        returns: numpy array of 48kHz audio
        """
        if isinstance(audio_data, bytes):
            # Convert bytes to numpy array (assuming 16-bit signed integers)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        else:
            audio_array = audio_data.astype(np.float32)
        
        # Upsample by inserting zeros
        upsampled = np.zeros(len(audio_array) * UPSAMPLE_FACTOR, dtype=np.float32)
        upsampled[::UPSAMPLE_FACTOR] = audio_array
        
        # Apply low-pass filter to interpolate
        filtered = signal.filtfilt(self.upsample_filter[0], self.upsample_filter[1], upsampled)
        
        # Scale to maintain amplitude
        filtered *= UPSAMPLE_FACTOR
        
        return filtered.astype(np.int16)
    
    def downsample_48k_to_8k(self, audio_data):
        """
        Downsample 48kHz audio to 8kHz for Vicidial
        audio_data: bytes or numpy array of 48kHz audio
        returns: numpy array of 8kHz audio
        """
        if isinstance(audio_data, bytes):
            # Convert bytes to numpy array (assuming 16-bit signed integers)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        else:
            audio_array = audio_data.astype(np.float32)
        
        # Apply anti-aliasing filter
        filtered = signal.filtfilt(self.downsample_filter[0], self.downsample_filter[1], audio_array)
        
        # Downsample by taking every 6th sample
        downsampled = filtered[::UPSAMPLE_FACTOR]
        
        return downsampled.astype(np.int16)

class LocalUltravoxClient:
    """Handles communication with local Ultravox model"""
    
    def __init__(self, model_path=ULTRAVOX_MODEL_PATH, device=ULTRAVOX_DEVICE):
        self.model_path = model_path
        self.device = device
        self.audio_processor = AudioProcessor()
        self.ultravox = None
        self.conversation_history = []
        self.is_initialized = False
        
    async def initialize_model(self):
        """Initialize the local Ultravox model"""
        try:
            logger.info(f"Loading Ultravox model from {self.model_path} on {self.device}")
            
            # Load the Ultravox model
            self.ultravox = Ultravox.from_pretrained(
                self.model_path,
                device_map=self.device,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.is_initialized = True
            logger.info("Ultravox model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Ultravox model: {e}")
            return False
    
    async def process_audio(self, audio_data, system_prompt="You are a helpful assistant."):
        """
        Process audio through local Ultravox model
        audio_data: 48kHz audio data as numpy array
        returns: 48kHz audio response as numpy array
        """
        if not self.is_initialized or self.ultravox is None:
            logger.error("Ultravox model not initialized")
            return None
        
        try:
            # Convert audio to the format expected by Ultravox
            # Ultravox expects audio as a tensor with shape (1, samples)
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
            
            # Process through Ultravox
            # Note: This is a simplified interface - you may need to adjust based on actual Ultravox API
            with torch.no_grad():
                response = self.ultravox.generate(
                    audio_tensor,
                    system_prompt=system_prompt,
                    max_new_tokens=100,  # Adjust as needed
                    temperature=0.7,
                    do_sample=True
                )
            
            # Extract audio response
            # The exact method depends on Ultravox's output format
            if hasattr(response, 'audio'):
                response_audio = response.audio
            else:
                # Fallback: generate silence if no audio response
                response_audio = torch.zeros_like(audio_tensor)
            
            # Convert back to numpy array
            response_array = response_audio.squeeze(0).cpu().numpy()
            
            return response_array.astype(np.int16)
            
        except Exception as e:
            logger.error(f"Error processing audio through Ultravox: {e}")
            return None
    
    async def create_conversation_session(self, system_prompt="You are a helpful assistant."):
        """Create a new conversation session"""
        if not self.is_initialized:
            await self.initialize_model()
        
        # Reset conversation history
        self.conversation_history = []
        
        logger.info(f"Created new conversation session with prompt: {system_prompt}")
        return True

class SecureLocalVicidialUltravoxBridge:
    """Secure bridge between Vicidial and local Ultravox model"""
    
    def __init__(self):
        self.ultravox_client = LocalUltravoxClient()
        self.audio_processor = AudioProcessor()
        self.security_manager = SecurityManager()
        self.active_connections = {}
        
    async def initialize(self):
        """Initialize the bridge and load the Ultravox model"""
        logger.info("Initializing Ultravox bridge...")
        
        # Initialize the Ultravox model
        if not await self.ultravox_client.initialize_model():
            raise Exception("Failed to initialize Ultravox model")
        
        logger.info("Ultravox bridge initialized successfully")
    
    async def handle_vicidial_connection(self, reader, writer):
        """Handle incoming Vicidial connection with authentication"""
        addr = writer.get_extra_info('peername')
        client_ip = addr[0]
        
        logger.info(f"New connection attempt from {addr}")
        
        # IP-based access control
        if not self.security_manager.verify_ip(client_ip):
            logger.warning(f"Connection rejected from unauthorized IP: {client_ip}")
            writer.close()
            await writer.wait_closed()
            return
        
        # Authentication (if enabled)
        if ENABLE_AUTH:
            auth_result = await self.authenticate_connection(reader, writer)
            if not auth_result:
                logger.warning(f"Authentication failed for {client_ip}")
                writer.close()
                await writer.wait_closed()
                return
        
        logger.info(f"Authenticated connection from {addr}")
        
        # Create new conversation session
        await self.ultravox_client.create_conversation_session()
        
        # Store connection info
        connection_id = f"{addr[0]}:{addr[1]}"
        self.active_connections[connection_id] = {
            'vicidial_reader': reader,
            'vicidial_writer': writer,
            'connected_at': time.time(),
            'audio_buffer': []
        }
        
        try:
            # Start bidirectional audio processing
            await asyncio.gather(
                self.process_vicidial_to_ultravox(connection_id),
                self.process_ultravox_to_vicidial(connection_id)
            )
        except Exception as e:
            logger.error(f"Error in connection {connection_id}: {e}")
        finally:
            await self.cleanup_connection(connection_id)
    
    async def authenticate_connection(self, reader, writer):
        """Authenticate incoming connection"""
        try:
            # Read authentication token (first 256 bytes)
            auth_data = await reader.read(256)
            if not auth_data:
                return False
            
            # Extract token (remove padding)
            token = auth_data.rstrip(b'\x00').decode('utf-8')
            
            # Verify token
            if self.security_manager.verify_token(token):
                # Send success response
                writer.write(b'AUTH_OK')
                await writer.drain()
                return True
            else:
                # Send failure response
                writer.write(b'AUTH_FAIL')
                await writer.drain()
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    async def process_vicidial_to_ultravox(self, connection_id):
        """Process audio from Vicidial (8kHz) to Ultravox (48kHz)"""
        conn = self.active_connections[connection_id]
        reader = conn['vicidial_reader']
        
        try:
            while True:
                # Read 8kHz audio from Vicidial
                audio_data = await reader.read(BYTES_PER_FRAME)
                if not audio_data:
                    break
                
                # Upsample to 48kHz
                upsampled_audio = self.audio_processor.upsample_8k_to_48k(audio_data)
                
                # Process through local Ultravox model
                response_audio = await self.ultravox_client.process_audio(upsampled_audio)
                
                if response_audio is not None:
                    # Store response for sending back to Vicidial
                    conn['audio_buffer'].append(response_audio)
                
        except Exception as e:
            logger.error(f"Error processing Vicidial to Ultravox: {e}")
    
    async def process_ultravox_to_vicidial(self, connection_id):
        """Process audio from Ultravox (48kHz) to Vicidial (8kHz)"""
        conn = self.active_connections[connection_id]
        writer = conn['vicidial_writer']
        
        try:
            while True:
                # Check if there's audio to send
                if conn['audio_buffer']:
                    ultravox_audio = conn['audio_buffer'].pop(0)
                    
                    # Downsample from 48kHz to 8kHz
                    downsampled_audio = self.audio_processor.downsample_48k_to_8k(ultravox_audio)
                    
                    # Send to Vicidial
                    writer.write(downsampled_audio.tobytes())
                    await writer.drain()
                else:
                    # No audio to send, wait a bit
                    await asyncio.sleep(0.01)  # 10ms
                    
        except Exception as e:
            logger.error(f"Error processing Ultravox to Vicidial: {e}")
    
    async def cleanup_connection(self, connection_id):
        """Clean up connection resources"""
        if connection_id in self.active_connections:
            conn = self.active_connections[connection_id]
            
            # Close Vicidial connection
            if conn['vicidial_writer']:
                conn['vicidial_writer'].close()
                await conn['vicidial_writer'].wait_closed()
            
            # Log connection duration
            duration = time.time() - conn['connected_at']
            logger.info(f"Connection {connection_id} closed after {duration:.2f} seconds")
            
            del self.active_connections[connection_id]
    
    async def start_server(self):
        """Start the secure bridge server"""
        # Initialize the bridge
        await self.initialize()
        
        server = await asyncio.start_server(
            self.handle_vicidial_connection,
            LISTEN_HOST,
            LISTEN_PORT
        )
        
        logger.info(f"Secure Local Ultravox bridge listening on {LISTEN_HOST}:{LISTEN_PORT}")
        logger.info(f"Using Ultravox model: {ULTRAVOX_MODEL_PATH}")
        logger.info(f"Device: {ULTRAVOX_DEVICE}")
        logger.info(f"Authentication enabled: {ENABLE_AUTH}")
        logger.info(f"Allowed IPs: {self.security_manager.allowed_ips if self.security_manager.allowed_ips else 'All IPs'}")
        
        async with server:
            await server.serve_forever()

async def main():
    """Main entry point"""
    bridge = SecureLocalVicidialUltravoxBridge()
    await bridge.start_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")

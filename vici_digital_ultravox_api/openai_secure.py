import asyncio
import base64
import json
import logging
import os
import socket
import struct
import threading
import aiohttp
import numpy as np
from array import array
import websockets
from scipy import signal
import hashlib
import hmac
import time

# Configuration
LISTEN_HOST, LISTEN_PORT = "0.0.0.0", 9092
ULTRAVOX_API_KEY = "IubPebty.dYGQig8TOpGz7ibA5iFuVFBwCA5kjohI"

# Security configuration
SECRET_KEY = os.getenv("BRIDGE_SECRET_KEY", "your-secret-key-change-this")
ALLOWED_IPS = os.getenv("ALLOWED_IPS", "").split(",")  # Comma-separated IPs
ENABLE_AUTH = os.getenv("ENABLE_AUTH", "true").lower() == "true"

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

class UltravoxClient:
    """Handles communication with Ultravox API"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.audio_processor = AudioProcessor()
    
    async def create_call(self, system_prompt="You are a helpful assistant..."):
        """Create a new Ultravox call session"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "systemPrompt": system_prompt,
                "model": "fixie-ai/ultravox",
                "voice": "Mark",
                "medium": {
                    "serverWebSocket": {
                        "inputSampleRate": ULTRAVOX_SAMPLE_RATE,
                        "outputSampleRate": ULTRAVOX_SAMPLE_RATE,
                    }
                }
            }
            
            headers = {
                'X-API-Key': self.api_key,
                'Content-Type': 'application/json'
            }
            
            async with session.post(
                'https://api.ultravox.ai/api/calls',
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('joinUrl')
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to create Ultravox call: {response.status} - {error_text}")
                    return None

class SecureVicidialUltravoxBridge:
    """Secure bridge between Vicidial and Ultravox with authentication"""
    
    def __init__(self):
        self.ultravox_client = UltravoxClient(ULTRAVOX_API_KEY)
        self.audio_processor = AudioProcessor()
        self.security_manager = SecurityManager()
        self.active_connections = {}
    
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
        
        # Create Ultravox call
        join_url = await self.ultravox_client.create_call()
        if not join_url:
            logger.error("Failed to create Ultravox call")
            writer.close()
            await writer.wait_closed()
            return
        
        # Connect to Ultravox WebSocket
        try:
            ultravox_ws = await websockets.connect(join_url)
            logger.info(f"Connected to Ultravox: {join_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Ultravox: {e}")
            writer.close()
            await writer.wait_closed()
            return
        
        # Store connection info
        connection_id = f"{addr[0]}:{addr[1]}"
        self.active_connections[connection_id] = {
            'vicidial_reader': reader,
            'vicidial_writer': writer,
            'ultravox_ws': ultravox_ws,
            'connected_at': time.time()
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
        ultravox_ws = conn['ultravox_ws']
        
        try:
            while True:
                # Read 8kHz audio from Vicidial
                audio_data = await reader.read(BYTES_PER_FRAME)
                if not audio_data:
                    break
                
                # Upsample to 48kHz
                upsampled_audio = self.audio_processor.upsample_8k_to_48k(audio_data)
                
                # Send to Ultravox
                await ultravox_ws.send(upsampled_audio.tobytes())
                
        except Exception as e:
            logger.error(f"Error processing Vicidial to Ultravox: {e}")
    
    async def process_ultravox_to_vicidial(self, connection_id):
        """Process audio from Ultravox (48kHz) to Vicidial (8kHz)"""
        conn = self.active_connections[connection_id]
        writer = conn['vicidial_writer']
        ultravox_ws = conn['ultravox_ws']
        
        try:
            async for message in ultravox_ws:
                if isinstance(message, bytes):
                    # Downsample from 48kHz to 8kHz
                    downsampled_audio = self.audio_processor.downsample_48k_to_8k(message)
                    
                    # Send to Vicidial
                    writer.write(downsampled_audio.tobytes())
                    await writer.drain()
                elif isinstance(message, str):
                    # Handle text messages (if any)
                    logger.info(f"Text message from Ultravox: {message}")
                    
        except Exception as e:
            logger.error(f"Error processing Ultravox to Vicidial: {e}")
    
    async def cleanup_connection(self, connection_id):
        """Clean up connection resources"""
        if connection_id in self.active_connections:
            conn = self.active_connections[connection_id]
            
            # Close Ultravox WebSocket
            if conn['ultravox_ws']:
                await conn['ultravox_ws'].close()
            
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
        server = await asyncio.start_server(
            self.handle_vicidial_connection,
            LISTEN_HOST,
            LISTEN_PORT
        )
        
        logger.info(f"Secure Vicidial-Ultravox bridge listening on {LISTEN_HOST}:{LISTEN_PORT}")
        logger.info(f"Authentication enabled: {ENABLE_AUTH}")
        logger.info(f"Allowed IPs: {self.security_manager.allowed_ips if self.security_manager.allowed_ips else 'All IPs'}")
        
        async with server:
            await server.serve_forever()

async def main():
    """Main entry point"""
    bridge = SecureVicidialUltravoxBridge()
    await bridge.start_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")





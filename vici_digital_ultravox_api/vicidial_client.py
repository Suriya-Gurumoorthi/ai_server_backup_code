import asyncio
import socket
import time
import hashlib
import hmac
import logging
import numpy as np
from typing import Optional, Callable, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltravoxBridgeClient:
    """Client for connecting to remote Ultravox bridge server"""
    
    def __init__(self, 
                 bridge_host: str, 
                 bridge_port: int = 9092,
                 secret_key: str = "your-secret-key-change-this",
                 enable_auth: bool = True):
        """
        Initialize Ultravox bridge client
        
        Args:
            bridge_host: IP address or hostname of the Ultravox bridge server
            bridge_port: Port number of the bridge server (default: 9092)
            secret_key: Secret key for authentication (must match bridge server)
            enable_auth: Whether to use authentication
        """
        self.bridge_host = bridge_host
        self.bridge_port = bridge_port
        self.secret_key = secret_key.encode('utf-8')
        self.enable_auth = enable_auth
        self.reader = None
        self.writer = None
        self.connected = False
        self.audio_callback = None
        
    def generate_auth_token(self) -> str:
        """Generate authentication token for bridge server"""
        timestamp = int(time.time())
        message = f"ultravox_bridge:{timestamp}".encode('utf-8')
        signature = hmac.new(self.secret_key, message, hashlib.sha256).hexdigest()
        return f"{timestamp}:{signature}"
    
    async def connect(self) -> bool:
        """Connect to the Ultravox bridge server"""
        try:
            logger.info(f"Connecting to Ultravox bridge at {self.bridge_host}:{self.bridge_port}")
            
            # Establish connection
            self.reader, self.writer = await asyncio.open_connection(
                self.bridge_host, 
                self.bridge_port
            )
            
            # Authenticate if enabled
            if self.enable_auth:
                auth_success = await self._authenticate()
                if not auth_success:
                    logger.error("Authentication failed")
                    await self.disconnect()
                    return False
            
            self.connected = True
            logger.info("Successfully connected to Ultravox bridge")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to bridge: {e}")
            await self.disconnect()
            return False
    
    async def _authenticate(self) -> bool:
        """Authenticate with the bridge server"""
        try:
            # Generate and send authentication token
            token = self.generate_auth_token()
            auth_data = token.encode('utf-8').ljust(256, b'\x00')  # Pad to 256 bytes
            self.writer.write(auth_data)
            await self.writer.drain()
            
            # Read response
            response = await self.reader.read(8)
            if response == b'AUTH_OK':
                logger.info("Authentication successful")
                return True
            else:
                logger.error("Authentication failed")
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the bridge server"""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        
        self.reader = None
        self.writer = None
        self.connected = False
        logger.info("Disconnected from Ultravox bridge")
    
    def set_audio_callback(self, callback: Callable[[bytes], None]):
        """Set callback function for receiving audio from Ultravox"""
        self.audio_callback = callback
    
    async def send_audio(self, audio_data: bytes):
        """Send audio data to Ultravox bridge"""
        if not self.connected or not self.writer:
            logger.error("Not connected to bridge")
            return False
        
        try:
            self.writer.write(audio_data)
            await self.writer.drain()
            return True
        except Exception as e:
            logger.error(f"Failed to send audio: {e}")
            return False
    
    async def receive_audio_loop(self):
        """Receive audio data from Ultravox bridge"""
        if not self.connected or not self.reader:
            logger.error("Not connected to bridge")
            return
        
        try:
            while self.connected:
                # Read audio data (320 bytes = 160 samples * 2 bytes per sample)
                audio_data = await self.reader.read(320)
                if not audio_data:
                    logger.info("Connection closed by bridge server")
                    break
                
                # Call audio callback if set
                if self.audio_callback:
                    self.audio_callback(audio_data)
                    
        except Exception as e:
            logger.error(f"Error receiving audio: {e}")
        finally:
            self.connected = False

class VicidialUltravoxIntegration:
    """Integration class for VICIdial to use Ultravox bridge"""
    
    def __init__(self, 
                 bridge_host: str,
                 bridge_port: int = 9092,
                 secret_key: str = "your-secret-key-change-this"):
        """
        Initialize VICIdial-Ultravox integration
        
        Args:
            bridge_host: IP address of the Ultravox bridge server
            bridge_port: Port number of the bridge server
            secret_key: Secret key for authentication
        """
        self.bridge_client = UltravoxBridgeClient(
            bridge_host=bridge_host,
            bridge_port=bridge_port,
            secret_key=secret_key
        )
        self.audio_queue = asyncio.Queue()
        self.is_running = False
        
    async def start(self):
        """Start the integration"""
        if self.is_running:
            return
        
        # Connect to bridge
        if not await self.bridge_client.connect():
            raise Exception("Failed to connect to Ultravox bridge")
        
        # Set audio callback
        self.bridge_client.set_audio_callback(self._handle_ultravox_audio)
        
        # Start audio processing
        self.is_running = True
        asyncio.create_task(self.bridge_client.receive_audio_loop())
        asyncio.create_task(self._audio_processing_loop())
        
        logger.info("VICIdial-Ultravox integration started")
    
    async def stop(self):
        """Stop the integration"""
        self.is_running = False
        await self.bridge_client.disconnect()
        logger.info("VICIdial-Ultravox integration stopped")
    
    async def send_vicidial_audio(self, audio_data: bytes):
        """Send audio from VICIdial to Ultravox"""
        if not self.is_running:
            logger.warning("Integration not running")
            return
        
        await self.bridge_client.send_audio(audio_data)
    
    def _handle_ultravox_audio(self, audio_data: bytes):
        """Handle audio received from Ultravox"""
        # Put audio in queue for processing
        asyncio.create_task(self.audio_queue.put(audio_data))
    
    async def _audio_processing_loop(self):
        """Process audio from Ultravox queue"""
        while self.is_running:
            try:
                # Get audio from queue
                audio_data = await asyncio.wait_for(
                    self.audio_queue.get(), 
                    timeout=1.0
                )
                
                # Process audio for VICIdial
                await self._process_ultravox_audio(audio_data)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
    
    async def _process_ultravox_audio(self, audio_data: bytes):
        """Process Ultravox audio for VICIdial"""
        # This is where you would integrate with VICIdial's audio system
        # For now, we'll just log that we received audio
        logger.debug(f"Received {len(audio_data)} bytes from Ultravox")
        
        # TODO: Implement VICIdial-specific audio processing
        # This might involve:
        # - Converting audio format if needed
        # - Sending to VICIdial's audio pipeline
        # - Handling call state changes
        pass

# Example usage and testing functions
async def test_connection(bridge_host: str, bridge_port: int = 9092):
    """Test connection to Ultravox bridge"""
    client = UltravoxBridgeClient(bridge_host, bridge_port)
    
    try:
        if await client.connect():
            logger.info("Connection test successful")
            
            # Send test audio (silence)
            test_audio = b'\x00' * 320
            await client.send_audio(test_audio)
            
            # Wait a bit
            await asyncio.sleep(2)
            
            await client.disconnect()
            return True
        else:
            logger.error("Connection test failed")
            return False
    except Exception as e:
        logger.error(f"Connection test error: {e}")
        return False

async def example_vicidial_integration():
    """Example of how to integrate with VICIdial"""
    # Configuration
    bridge_host = "192.168.1.100"  # Replace with your bridge server IP
    bridge_port = 9092
    secret_key = "your-secret-key-change-this"
    
    # Create integration
    integration = VicidialUltravoxIntegration(
        bridge_host=bridge_host,
        bridge_port=bridge_port,
        secret_key=secret_key
    )
    
    try:
        # Start integration
        await integration.start()
        
        # Simulate VICIdial sending audio
        for i in range(10):
            # Generate some test audio (sine wave)
            sample_rate = 8000
            duration = 0.02  # 20ms
            frequency = 440  # A4 note
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * frequency * t) * 32767
            audio_bytes = audio.astype(np.int16).tobytes()
            
            await integration.send_vicidial_audio(audio_bytes)
            await asyncio.sleep(0.02)
        
        # Wait for processing
        await asyncio.sleep(5)
        
    finally:
        await integration.stop()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test connection
        bridge_host = sys.argv[1]
        asyncio.run(test_connection(bridge_host))
    else:
        # Run example
        asyncio.run(example_vicidial_integration())





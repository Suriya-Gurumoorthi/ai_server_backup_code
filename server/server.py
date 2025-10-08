"""
Main server module for the Ultravox WebSocket server.
This is the entry point that orchestrates all components.
"""

import asyncio
import logging
import websockets

from config import (
    SERVER_HOST, SERVER_PORT, MAX_MESSAGE_SIZE, PING_INTERVAL, 
    PING_TIMEOUT, CLOSE_TIMEOUT, DEVICE
)
from utils import setup_logging
from models import model_manager
from websocket_handler import websocket_handler


class UltravoxServer:
    """Main server class that orchestrates all components."""
    
    def __init__(self):
        self.logger = setup_logging()
        self.server = None
    
    async def start(self):
        """Start the WebSocket server."""
        try:
            # Configure server with better connection handling
            self.server = await websockets.serve(
                websocket_handler.handle_connection,
                SERVER_HOST,
                SERVER_PORT,
                max_size=MAX_MESSAGE_SIZE,
                ping_interval=PING_INTERVAL,
                ping_timeout=PING_TIMEOUT,
                close_timeout=CLOSE_TIMEOUT,
                compression=None  # Disable compression for better performance
            )
            
            self._log_server_info()
            self.logger.info("Ultravox WebSocket server started successfully!")
            
            # Keep server running
            await self.server.wait_closed()
            
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            raise
        finally:
            await self.shutdown()
    
    def _log_server_info(self):
        """Log server configuration and status information."""
        self.logger.info("Ultravox WebSocket server started on port 8000")
        self.logger.info(f"Server configuration:")
        self.logger.info(f"  - Host: {SERVER_HOST}")
        self.logger.info(f"  - Port: {SERVER_PORT}")
        self.logger.info(f"  - Max message size: {MAX_MESSAGE_SIZE // 1024 // 1024}MB")
        self.logger.info(f"  - Ping interval: {PING_INTERVAL}s")
        self.logger.info(f"  - Ping timeout: {PING_TIMEOUT}s")
        self.logger.info(f"  - Compression: disabled")
        self.logger.info(f"  - Model device: {DEVICE}")
        if DEVICE == "cuda":
            self.logger.info(f"  - GPU acceleration: enabled")
        self.logger.info(f"  - TTS: {'enabled' if model_manager.is_tts_available() else 'disabled'}")
    
    async def shutdown(self):
        """Shutdown the server and clean up resources."""
        self.logger.info("Shutting down server...")
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        # Shutdown model manager
        model_manager.shutdown()
        self.logger.info("Server shutdown complete")


async def main():
    """Main entry point for the server."""
    server = UltravoxServer()
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())

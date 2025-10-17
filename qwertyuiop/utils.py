"""
Utility functions and logging configuration for the Ultravox WebSocket server.
"""

import logging
import asyncio
import websockets
from typing import Any, Dict
from config import LOG_LEVEL


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


async def safe_send_response(websocket, message, client_address):
    """Safely send a response, handling connection state properly."""
    try:
        await websocket.send(message)
        logger = logging.getLogger(__name__)
        logger.info(f"Response sent successfully to {client_address}")
        return True
    except websockets.ConnectionClosed:
        logger = logging.getLogger(__name__)
        logger.warning(f"Connection to {client_address} closed while sending response")
        return False
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error sending response to {client_address}: {e}")
        return False


def get_connection_info(websocket) -> str:
    """Get formatted client address information."""
    return f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"


def validate_text_for_tts(text: str) -> str:
    """Validate and clean text for TTS processing."""
    if not text or not text.strip():
        return ""
    
    # Truncate very long text
    if len(text) > 1000:
        return text[:1000]
    
    return text.strip()

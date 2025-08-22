"""
Ollama Client for communicating with local Ollama instance
"""

import requests
import json
import logging
from typing import Dict, Any, Optional
from config import OLLAMA_URL, MODEL_NAME

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self):
        self.base_url = OLLAMA_URL
        self.model = MODEL_NAME
    
    def send_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Send a prompt to Ollama and return the response
        
        Args:
            prompt (str): The user's prompt
            
        Returns:
            Dict[str, Any]: Formatted response with 'response' field
            
        Raises:
            requests.RequestException: If Ollama is not available
        """
        try:
            # Prepare the request to Ollama
            ollama_request = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            # Send request to Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=ollama_request,
                timeout=30  # 30 second timeout
            )
            
            # Check if request was successful
            response.raise_for_status()
            
            # Parse Ollama response
            ollama_response = response.json()
            
            # Extract the response text
            response_text = ollama_response.get("response", "")
            
            # Return simplified format
            return {
                "response": response_text
            }
            
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to Ollama - service may not be running")
            raise requests.RequestException("Ollama service not available")
            
        except requests.exceptions.Timeout:
            logger.error("Request to Ollama timed out")
            raise requests.RequestException("Ollama request timed out")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to Ollama failed: {e}")
            raise requests.RequestException("Ollama service not available")
            
        except json.JSONDecodeError:
            logger.error("Invalid JSON response from Ollama")
            raise requests.RequestException("Invalid response from Ollama")
            
        except Exception as e:
            logger.error(f"Unexpected error communicating with Ollama: {e}")
            raise requests.RequestException("Ollama service not available")
    
    def is_available(self) -> bool:
        """
        Check if Ollama is available and responding
        
        Returns:
            bool: True if Ollama is available, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

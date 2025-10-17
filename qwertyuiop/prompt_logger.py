"""
Prompt Logging Module for Ultravox WebSocket Server.
Handles logging of all prompts sent to AI models for debugging and analysis.
"""

import json
import os
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class PromptLogger:
    """Manages prompt logging for AI model interactions."""
    
    def __init__(self, storage_dir: str = "prompt_logs"):
        """
        Initialize the PromptLogger.
        
        Args:
            storage_dir: Directory to store prompt log files
        """
        self.logger = logging.getLogger(__name__)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Active prompt sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info(f"PromptLogger initialized with storage directory: {self.storage_dir}")
    
    def create_session(self, connection_id: str, session_id: str = None) -> str:
        """
        Create a new prompt logging session.
        
        Args:
            connection_id: Unique connection identifier
            session_id: Optional existing session ID to link with
            
        Returns:
            prompt_session_id: Unique prompt session identifier
        """
        prompt_session_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        session_data = {
            "prompt_session_id": prompt_session_id,
            "connection_id": connection_id,
            "linked_session_id": session_id,
            "created_at": timestamp,
            "prompt_logs": [],
            "file_path": self.storage_dir / f"prompts_{timestamp}_{prompt_session_id[:8]}.json"
        }
        
        self.active_sessions[connection_id] = session_data
        
        self.logger.info(f"Created new prompt session {prompt_session_id} for connection {connection_id}")
        return prompt_session_id
    
    def log_prompt(self, connection_id: str, prompt_type: str, model_name: str, 
                   input_data: Any, response: str = None, metadata: Dict[str, Any] = None) -> bool:
        """
        Log a prompt sent to an AI model.
        
        Args:
            connection_id: Connection identifier
            prompt_type: Type of prompt (e.g., 'transcribe', 'process_audio', 'summarize')
            model_name: Name of the model being used
            input_data: The input data/prompt sent to the model
            response: The response from the model (optional)
            metadata: Additional metadata about the prompt
            
        Returns:
            bool: True if successful, False otherwise
        """
        if connection_id not in self.active_sessions:
            self.logger.warning(f"No active prompt session found for connection {connection_id}")
            return False
        
        session_data = self.active_sessions[connection_id]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create prompt log entry
        prompt_entry = {
            "prompt_type": prompt_type,
            "model_name": model_name,
            "timestamp": timestamp,
            "input_data": input_data,
            "response": response,
            "metadata": metadata or {}
        }
        
        session_data["prompt_logs"].append(prompt_entry)
        
        # Save to file
        self._save_prompts_to_file(session_data)
        
        self.logger.info(f"Logged prompt for connection {connection_id}: {prompt_type} -> {model_name}")
        return True
    
    def log_ultravox_prompt(self, connection_id: str, audio_data: bytes, 
                           conversation_turns: List[Dict[str, str]], 
                           response: str = None) -> bool:
        """Log an Ultravox prompt with audio and conversation context."""
        metadata = {
            "audio_length": len(audio_data),
            "conversation_turns_count": len(conversation_turns),
            "turns": conversation_turns
        }
        
        return self.log_prompt(
            connection_id=connection_id,
            prompt_type="ultravox_audio_processing",
            model_name="ultravox-v0_5-llama-3_2-1b",
            input_data={
                "audio_data_size": len(audio_data),
                "conversation_turns": conversation_turns
            },
            response=response,
            metadata=metadata
        )
    
    def log_whisper_prompt(self, connection_id: str, audio_data: bytes, 
                          response: str = None) -> bool:
        """Log a Whisper transcription prompt."""
        metadata = {
            "audio_length": len(audio_data),
            "model": "openai/whisper-small"
        }
        
        return self.log_prompt(
            connection_id=connection_id,
            prompt_type="whisper_transcription",
            model_name="whisper-small",
            input_data={
                "audio_data_size": len(audio_data)
            },
            response=response,
            metadata=metadata
        )
    
    def log_summarization_prompt(self, connection_id: str, conversation_history: List[Dict[str, Any]], 
                                response: str = None) -> bool:
        """Log a conversation summarization prompt."""
        metadata = {
            "conversation_entries": len(conversation_history),
            "history": conversation_history
        }
        
        return self.log_prompt(
            connection_id=connection_id,
            prompt_type="conversation_summarization",
            model_name="ultravox-v0_5-llama-3_2-1b",
            input_data={
                "conversation_history": conversation_history
            },
            response=response,
            metadata=metadata
        )
    
    def get_prompt_logs(self, connection_id: str) -> List[Dict[str, Any]]:
        """
        Get the prompt logs for a specific connection.
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            List of prompt log entries
        """
        if connection_id not in self.active_sessions:
            self.logger.warning(f"No active prompt session found for connection {connection_id}")
            return []
        
        return self.active_sessions[connection_id]["prompt_logs"]
    
    def _save_prompts_to_file(self, session_data: Dict[str, Any]) -> bool:
        """
        Save prompt logs to JSON file.
        
        Args:
            session_data: Session data containing prompt logs
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Prepare data for JSON serialization
            save_data = {
                "prompt_session_id": session_data["prompt_session_id"],
                "connection_id": session_data["connection_id"],
                "linked_session_id": session_data.get("linked_session_id"),
                "created_at": session_data["created_at"],
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "prompt_logs": session_data["prompt_logs"]
            }
            
            # Write to JSON file
            with open(session_data["file_path"], 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Saved prompt logs to {session_data['file_path']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save prompt logs to file: {e}")
            return False
    
    def load_prompts_from_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load prompt logs from JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Loaded prompt data or None if failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            self.logger.error(f"Failed to load prompt logs from {file_path}: {e}")
            return None
    
    def end_session(self, connection_id: str) -> bool:
        """
        End a prompt logging session and clean up.
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        if connection_id not in self.active_sessions:
            self.logger.warning(f"No active prompt session found for connection {connection_id}")
            return False
        
        session_data = self.active_sessions[connection_id]
        
        # Final save
        self._save_prompts_to_file(session_data)
        
        # Log session summary
        logs_count = len(session_data["prompt_logs"])
        self.logger.info(f"Ended prompt session {session_data['prompt_session_id']} for connection {connection_id} with {logs_count} prompt logs")
        
        # Remove from active sessions
        del self.active_sessions[connection_id]
        
        return True
    
    def get_session_summary(self, connection_id: str) -> Dict[str, Any]:
        """
        Get a summary of the current prompt session.
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            Session summary dictionary
        """
        if connection_id not in self.active_sessions:
            return {"error": "No active prompt session found"}
        
        session_data = self.active_sessions[connection_id]
        logs = session_data["prompt_logs"]
        
        # Count different types of prompts
        prompt_types = {}
        for log in logs:
            prompt_type = log.get("prompt_type", "unknown")
            prompt_types[prompt_type] = prompt_types.get(prompt_type, 0) + 1
        
        return {
            "prompt_session_id": session_data["prompt_session_id"],
            "created_at": session_data["created_at"],
            "total_prompts": len(logs),
            "prompt_types": prompt_types,
            "file_path": str(session_data["file_path"])
        }
    
    def cleanup_old_logs(self, days_old: int = 7) -> int:
        """
        Clean up old prompt log files.
        
        Args:
            days_old: Number of days after which to delete files
            
        Returns:
            Number of files cleaned up
        """
        import time
        
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        cleaned_count = 0
        
        try:
            for file_path in self.storage_dir.glob("prompts_*.json"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    cleaned_count += 1
                    self.logger.info(f"Cleaned up old prompt log file: {file_path}")
        
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
        
        return cleaned_count


# Global prompt logger instance
prompt_logger = PromptLogger()

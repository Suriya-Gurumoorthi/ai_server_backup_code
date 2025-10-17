"""
Transcription Management Module for Ultravox WebSocket Server.
Handles conversation history, transcription storage, and context management.
"""

import json
import os
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class TranscriptionManager:
    """Manages conversation transcriptions and history storage."""
    
    def __init__(self, storage_dir: str = "conversation_history"):
        """
        Initialize the TranscriptionManager.
        
        Args:
            storage_dir: Directory to store conversation history files
        """
        self.logger = logging.getLogger(__name__)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Active conversation sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Session state for name management
        self.session_states: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info(f"TranscriptionManager initialized with storage directory: {self.storage_dir}")
    
    def create_session(self, connection_id: str) -> str:
        """
        Create a new conversation session.
        
        Args:
            connection_id: Unique connection identifier
            
        Returns:
            session_id: Unique session identifier
        """
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        session_data = {
            "session_id": session_id,
            "connection_id": connection_id,
            "created_at": timestamp,
            "conversation_history": [],
            "file_path": self.storage_dir / f"conversation_{timestamp}_{session_id[:8]}.json"
        }
        
        self.active_sessions[connection_id] = session_data
        
        # Initialize session state for name management
        self.session_states[connection_id] = {
            "candidate_name": None,
            "recruiter_name": "Alexa",  # Fixed recruiter name
            "name_confirmed": False
        }
        
        # Don't create the file immediately - it will be created when first conversation entry is added
        self.logger.info(f"Created new session {session_id} for connection {connection_id} (file will be created on first message)")
        return session_id
    
    def add_user_transcription(self, connection_id: str, transcription: str) -> bool:
        """
        Add user transcription to the conversation history.
        
        Args:
            connection_id: Connection identifier
            transcription: User's transcribed speech
            
        Returns:
            bool: True if successful, False otherwise
        """
        if connection_id not in self.active_sessions:
            self.logger.warning(f"No active session found for connection {connection_id}")
            return False
        
        session_data = self.active_sessions[connection_id]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add user transcription to history
        user_entry = {
            "User": transcription,
            "timestamp": timestamp,
            "type": "user"
        }
        
        session_data["conversation_history"].append(user_entry)
        
        # Save to file (this will create the file on first entry)
        self._save_conversation_to_file(session_data)
        
        self.logger.info(f"Added user transcription for connection {connection_id}: {transcription[:50]}...")
        return True
    
    def add_ai_transcription(self, connection_id: str, transcription: str) -> bool:
        """
        Add AI transcription to the conversation history.
        
        Args:
            connection_id: Connection identifier
            transcription: AI's transcribed speech
            
        Returns:
            bool: True if successful, False otherwise
        """
        if connection_id not in self.active_sessions:
            self.logger.warning(f"No active session found for connection {connection_id}")
            return False
        
        session_data = self.active_sessions[connection_id]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add AI transcription to history
        ai_entry = {
            "AI": transcription,
            "timestamp": timestamp,
            "type": "ai"
        }
        
        session_data["conversation_history"].append(ai_entry)
        
        # Save to file
        self._save_conversation_to_file(session_data)
        
        self.logger.info(f"Added AI transcription for connection {connection_id}: {transcription[:50]}...")
        return True
    
    def get_conversation_history(self, connection_id: str) -> List[Dict[str, Any]]:
        """
        Get the conversation history for a specific connection.
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            List of conversation entries
        """
        if connection_id not in self.active_sessions:
            self.logger.warning(f"No active session found for connection {connection_id}")
            return []
        
        return self.active_sessions[connection_id]["conversation_history"]
    
    def get_conversation_context_for_ai(self, connection_id: str) -> List[Dict[str, str]]:
        """
        Get conversation history formatted for AI model context.
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            List of conversation turns formatted for AI model
        """
        if connection_id not in self.active_sessions:
            self.logger.warning(f"No active session found for connection {connection_id}")
            return []
        
        conversation_history = self.active_sessions[connection_id]["conversation_history"]
        ai_context = []
        
        for entry in conversation_history:
            if entry["type"] == "user":
                # Clean user content to remove any speaker prefixes or identifiers
                user_content = entry["User"].strip()
                # Remove any "Candidate:" or similar prefixes that might cause confusion
                if user_content.startswith(("Candidate:", "User:", "I am", "My name is")):
                    # Extract just the actual content after the prefix
                    parts = user_content.split(":", 1)
                    if len(parts) > 1:
                        user_content = parts[1].strip()
                
                ai_context.append({
                    "role": "user",
                    "content": user_content
                })
            elif entry["type"] == "ai":
                # Clean AI content to ensure consistent identity
                ai_content = entry["AI"].strip()
                # Remove any inconsistent name references that might confuse the model
                # Replace any "I'm Alex" or "I'm Rachel" with "I'm Alexa" to maintain consistency
                import re
                ai_content = re.sub(r"I'm (Alex|Rachel|Emily|Alexa)(?![a-z])", "I'm Alexa", ai_content, flags=re.IGNORECASE)
                ai_content = re.sub(r"My name is (Alex|Rachel|Emily|Alexa)(?![a-z])", "My name is Alexa", ai_content, flags=re.IGNORECASE)
                
                ai_context.append({
                    "role": "assistant", 
                    "content": ai_content
                })
        
        return ai_context
    
    def update_candidate_name(self, connection_id: str, name: str) -> bool:
        """
        Update the candidate's name when they provide it.
        
        Args:
            connection_id: Connection identifier
            name: Candidate's name
            
        Returns:
            bool: True if successful, False otherwise
        """
        if connection_id not in self.session_states:
            self.logger.warning(f"No session state found for connection {connection_id}")
            return False
        
        self.session_states[connection_id]["candidate_name"] = name
        self.session_states[connection_id]["name_confirmed"] = True
        self.logger.info(f"Updated candidate name for connection {connection_id}: {name}")
        return True
    
    def get_session_state(self, connection_id: str) -> Dict[str, Any]:
        """
        Get the current session state including names.
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            Session state dictionary
        """
        if connection_id not in self.session_states:
            return {"candidate_name": None, "recruiter_name": "Alexa", "name_confirmed": False}
        
        return self.session_states[connection_id]
    
    def _save_conversation_to_file(self, session_data: Dict[str, Any]) -> bool:
        """
        Save conversation history to JSON file.
        
        Args:
            session_data: Session data containing conversation history
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Prepare data for JSON serialization
            save_data = {
                "session_id": session_data["session_id"],
                "connection_id": session_data["connection_id"],
                "created_at": session_data["created_at"],
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "conversation_history": session_data["conversation_history"]
            }
            
            # Write to JSON file
            with open(session_data["file_path"], 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Saved conversation to {session_data['file_path']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save conversation to file: {e}")
            return False
    
    def load_conversation_from_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load conversation history from JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Loaded conversation data or None if failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            self.logger.error(f"Failed to load conversation from {file_path}: {e}")
            return None
    
    async def generate_and_save_summary(self, connection_id: str):
        """Generate call summary and save it to the session file."""
        if connection_id not in self.active_sessions:
            self.logger.warning(f"No active session for summary generation: {connection_id}")
            return False

        session_data = self.active_sessions[connection_id]
        history = session_data["conversation_history"]

        # Defensive check: ensure history is a list of dicts
        if not isinstance(history, list):
            self.logger.error(f"Conversation history is not a list, got {type(history)}: {history}")
            return False
        
        if not all(isinstance(entry, dict) for entry in history):
            self.logger.error(f"Not all conversation entries are dictionaries")
            return False

        from models import model_manager  # Import here to avoid circular import

        try:
            summary = await model_manager.summarize_conversation(history, connection_id)
            # Load existing JSON
            file_path = session_data["file_path"]
            with open(file_path, "r+", encoding="utf-8") as f:
                session_json = json.load(f)
                session_json["summary"] = summary
                f.seek(0)
                json.dump(session_json, f, indent=2, ensure_ascii=False)
                f.truncate()
            self.logger.info(f"Summary saved for session {session_data['session_id']}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to generate/save summary: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    async def end_session(self, connection_id: str) -> bool:
        """
        End a conversation session and clean up.
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        if connection_id not in self.active_sessions:
            self.logger.warning(f"No active session found for connection {connection_id}")
            return False
        
        session_data = self.active_sessions[connection_id]
        
        # Final save
        self._save_conversation_to_file(session_data)
        
        # Run and save summary
        await self.generate_and_save_summary(connection_id)
        
        # Log session summary
        history_count = len(session_data["conversation_history"])
        self.logger.info(f"Ended session {session_data['session_id']} for connection {connection_id} with {history_count} entries")
        
        # Remove from active sessions and session states
        del self.active_sessions[connection_id]
        if connection_id in self.session_states:
            del self.session_states[connection_id]
        
        return True
    
    def get_session_summary(self, connection_id: str) -> Dict[str, Any]:
        """
        Get a summary of the current session.
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            Session summary dictionary
        """
        if connection_id not in self.active_sessions:
            return {"error": "No active session found"}
        
        session_data = self.active_sessions[connection_id]
        history = session_data["conversation_history"]
        
        user_entries = len([entry for entry in history if entry["type"] == "user"])
        ai_entries = len([entry for entry in history if entry["type"] == "ai"])
        
        return {
            "session_id": session_data["session_id"],
            "created_at": session_data["created_at"],
            "total_entries": len(history),
            "user_entries": user_entries,
            "ai_entries": ai_entries,
            "file_path": str(session_data["file_path"])
        }
    
    def cleanup_old_sessions(self, days_old: int = 7) -> int:
        """
        Clean up old conversation files.
        
        Args:
            days_old: Number of days after which to delete files
            
        Returns:
            Number of files cleaned up
        """
        import time
        
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        cleaned_count = 0
        
        try:
            for file_path in self.storage_dir.glob("conversation_*.json"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    cleaned_count += 1
                    self.logger.info(f"Cleaned up old conversation file: {file_path}")
        
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
        
        return cleaned_count


# Global transcription manager instance
transcription_manager = TranscriptionManager()


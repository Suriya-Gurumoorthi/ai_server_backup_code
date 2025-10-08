"""
Call Recording Module for Ultravox WebSocket API

This module handles recording and storing call transcripts for each conversation.
It creates separate files for each call session and logs both user and AI responses.

Features:
- Automatic file creation for each call session
- User transcript logging (from audio transcription)
- AI response logging
- Timestamped entries
- JSON format for easy parsing
- Call session management
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class CallEntry:
    """Represents a single entry in the call transcript"""
    timestamp: str
    speaker: str  # "user" or "ai"
    content: str
    turn_number: int
    metadata: Optional[Dict] = None

@dataclass
class CallSession:
    """Represents a complete call session"""
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    entries: List[CallEntry] = None
    total_turns: int = 0
    
    def __post_init__(self):
        if self.entries is None:
            self.entries = []

class CallRecorder:
    """Manages call recording and transcript storage"""
    
    def __init__(self, recordings_dir: str = "call_recordings"):
        """
        Initialize the call recorder
        
        Args:
            recordings_dir: Directory to store call recordings
        """
        self.recordings_dir = recordings_dir
        self.active_sessions: Dict[str, CallSession] = {}
        
        # Ensure recordings directory exists
        os.makedirs(recordings_dir, exist_ok=True)
        logger.info(f"Call recorder initialized with directory: {recordings_dir}")
    
    def start_call_session(self, session_id: str) -> CallSession:
        """
        Start a new call session
        
        Args:
            session_id: Unique identifier for the call session
            
        Returns:
            CallSession object for the new session
        """
        start_time = datetime.now().isoformat()
        session = CallSession(
            session_id=session_id,
            start_time=start_time
        )
        
        self.active_sessions[session_id] = session
        logger.info(f"📞 Started call session: {session_id} at {start_time}")
        
        return session
    
    def end_call_session(self, session_id: str) -> Optional[str]:
        """
        End a call session and save the transcript
        
        Args:
            session_id: Session identifier
            
        Returns:
            Path to the saved transcript file, or None if session not found
        """
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found in active sessions")
            return None
        
        session = self.active_sessions[session_id]
        session.end_time = datetime.now().isoformat()
        session.total_turns = len(session.entries)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"call_{session_id}_{timestamp}.json"
        filepath = os.path.join(self.recordings_dir, filename)
        
        # Save transcript to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(asdict(session), f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved call transcript: {filepath}")
            logger.info(f"📊 Call summary: {session.total_turns} turns, duration: {self._calculate_duration(session)}")
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving call transcript: {e}")
            return None
    
    def log_user_message(self, session_id: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """
        Log a user message/transcript
        
        Args:
            session_id: Session identifier
            content: User message content
            metadata: Optional metadata
            
        Returns:
            True if logged successfully, False otherwise
        """
        return self._log_entry(session_id, "user", content, metadata)
    
    def log_ai_response(self, session_id: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """
        Log an AI response
        
        Args:
            session_id: Session identifier
            content: AI response content
            metadata: Optional metadata
            
        Returns:
            True if logged successfully, False otherwise
        """
        return self._log_entry(session_id, "ai", content, metadata)
    
    def _log_entry(self, session_id: str, speaker: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """
        Log an entry to the call session
        
        Args:
            session_id: Session identifier
            speaker: "user" or "ai"
            content: Message content
            metadata: Optional metadata
            
        Returns:
            True if logged successfully, False otherwise
        """
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found, creating new session")
            self.start_call_session(session_id)
        
        session = self.active_sessions[session_id]
        
        # Create new entry
        entry = CallEntry(
            timestamp=datetime.now().isoformat(),
            speaker=speaker,
            content=content,
            turn_number=len(session.entries) + 1,
            metadata=metadata or {}
        )
        
        session.entries.append(entry)
        
        logger.info(f"📝 Logged {speaker} message in session {session_id}: {content[:50]}...")
        return True
    
    def _calculate_duration(self, session: CallSession) -> str:
        """Calculate call duration"""
        if not session.start_time or not session.end_time:
            return "Unknown"
        
        try:
            start = datetime.fromisoformat(session.start_time)
            end = datetime.fromisoformat(session.end_time)
            duration = end - start
            return str(duration)
        except Exception:
            return "Unknown"
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get information about a call session"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        return {
            "session_id": session.session_id,
            "start_time": session.start_time,
            "total_turns": len(session.entries),
            "duration": self._calculate_duration(session) if session.end_time else "Ongoing"
        }
    
    def list_recorded_calls(self) -> List[Dict]:
        """List all recorded call files"""
        try:
            files = []
            for filename in os.listdir(self.recordings_dir):
                if filename.startswith("call_") and filename.endswith(".json"):
                    filepath = os.path.join(self.recordings_dir, filename)
                    stat = os.stat(filepath)
                    files.append({
                        "filename": filename,
                        "filepath": filepath,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
            
            # Sort by modification time (newest first)
            files.sort(key=lambda x: x["modified"], reverse=True)
            return files
            
        except Exception as e:
            logger.error(f"Error listing recorded calls: {e}")
            return []
    
    def get_call_transcript(self, filepath: str) -> Optional[Dict]:
        """Load a call transcript from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading call transcript: {e}")
            return None
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old active sessions that might be stuck"""
        current_time = datetime.now()
        sessions_to_remove = []
        
        for session_id, session in self.active_sessions.items():
            try:
                start_time = datetime.fromisoformat(session.start_time)
                age_hours = (current_time - start_time).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    sessions_to_remove.append(session_id)
                    logger.warning(f"Cleaning up old session {session_id} (age: {age_hours:.1f} hours)")
            except Exception as e:
                logger.error(f"Error checking session age: {e}")
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            self.end_call_session(session_id)

# Global instance
call_recorder = None

def initialize_call_recorder(recordings_dir: str = "call_recordings") -> CallRecorder:
    """Initialize the global call recorder instance"""
    global call_recorder
    if call_recorder is None:
        call_recorder = CallRecorder(recordings_dir)
    return call_recorder

def get_call_recorder() -> Optional[CallRecorder]:
    """Get the global call recorder instance"""
    return call_recorder

# Convenience functions for easy integration
def start_call_session(session_id: str) -> CallSession:
    """Start a new call session"""
    if call_recorder is None:
        logger.warning("Call recorder not initialized")
        return None
    return call_recorder.start_call_session(session_id)

def end_call_session(session_id: str) -> Optional[str]:
    """End a call session and save transcript"""
    if call_recorder is None:
        logger.warning("Call recorder not initialized")
        return None
    return call_recorder.end_call_session(session_id)

def log_user_message(session_id: str, content: str, metadata: Optional[Dict] = None) -> bool:
    """Log a user message"""
    if call_recorder is None:
        logger.warning("Call recorder not initialized")
        return False
    return call_recorder.log_user_message(session_id, content, metadata)

def log_ai_response(session_id: str, content: str, metadata: Optional[Dict] = None) -> bool:
    """Log an AI response"""
    if call_recorder is None:
        logger.warning("Call recorder not initialized")
        return False
    return call_recorder.log_ai_response(session_id, content, metadata)

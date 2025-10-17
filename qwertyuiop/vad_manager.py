"""
Voice Activity Detection Manager for Ultravox Server

Manages VAD instances for each connection and provides integration with the WebSocket handler.
"""

import logging
from typing import Dict, Optional, Any
from voice_activity_detection import VoiceActivityDetector
from config import (
    VAD_ENABLED, VAD_ENERGY_THRESHOLD, VAD_SILENCE_DURATION_MS, VAD_MIN_SPEECH_DURATION_MS,
    VAD_HIGH_PASS_CUTOFF, VAD_MIN_CONSECUTIVE_FRAMES, VAD_SPECTRAL_FLATNESS_THRESHOLD,
    VAD_BARGE_IN_CONSECUTIVE_FRAMES, VAD_BARGE_IN_ENABLED, VAD_DEBUG_LOGGING,
    AUDIO_SAMPLE_RATE
)

logger = logging.getLogger(__name__)

class VADManager:
    """Manages Voice Activity Detection for multiple connections."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.vad_instances: Dict[str, VoiceActivityDetector] = {}
        self.enabled = VAD_ENABLED
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'speech_segments_detected': 0,
            'silence_chunks_filtered': 0,
            'barge_in_events': 0
        }
        
        if self.enabled:
            self.logger.info("VAD Manager initialized with enhanced voice activity detection")
            self.logger.info(f"VAD Configuration:")
            self.logger.info(f"  - Energy threshold: {VAD_ENERGY_THRESHOLD}")
            self.logger.info(f"  - Silence duration: {VAD_SILENCE_DURATION_MS}ms")
            self.logger.info(f"  - Min speech duration: {VAD_MIN_SPEECH_DURATION_MS}ms")
            self.logger.info(f"  - High-pass cutoff: {VAD_HIGH_PASS_CUTOFF}Hz")
            self.logger.info(f"  - Consecutive frames: {VAD_MIN_CONSECUTIVE_FRAMES}")
            self.logger.info(f"  - Barge-in enabled: {VAD_BARGE_IN_ENABLED}")
        else:
            self.logger.info("VAD Manager initialized but VAD is disabled")
    
    def create_vad_for_connection(self, connection_id: str) -> Optional[VoiceActivityDetector]:
        """
        Create a new VAD instance for a connection.
        
        Args:
            connection_id: Unique identifier for the connection
            
        Returns:
            VoiceActivityDetector instance or None if VAD is disabled
        """
        if not self.enabled:
            return None
        
        try:
            vad = VoiceActivityDetector(
                energy_threshold=VAD_ENERGY_THRESHOLD,
                silence_duration_ms=VAD_SILENCE_DURATION_MS,
                min_speech_duration_ms=VAD_MIN_SPEECH_DURATION_MS,
                sample_rate=AUDIO_SAMPLE_RATE,
                high_pass_cutoff=VAD_HIGH_PASS_CUTOFF,
                min_consecutive_frames=VAD_MIN_CONSECUTIVE_FRAMES,
                spectral_flatness_threshold=VAD_SPECTRAL_FLATNESS_THRESHOLD,
                debug_logging=VAD_DEBUG_LOGGING
            )
            
            # Configure barge-in detection if enabled
            if VAD_BARGE_IN_ENABLED:
                vad.barge_in_consecutive_frames_threshold = VAD_BARGE_IN_CONSECUTIVE_FRAMES
            
            self.vad_instances[connection_id] = vad
            self.stats['total_connections'] += 1
            self.stats['active_connections'] += 1
            
            self.logger.info(f"Created VAD instance for connection {connection_id}")
            return vad
            
        except Exception as e:
            self.logger.error(f"Failed to create VAD instance for connection {connection_id}: {e}")
            return None
    
    def get_vad_for_connection(self, connection_id: str) -> Optional[VoiceActivityDetector]:
        """
        Get VAD instance for a connection.
        
        Args:
            connection_id: Unique identifier for the connection
            
        Returns:
            VoiceActivityDetector instance or None
        """
        return self.vad_instances.get(connection_id)
    
    def remove_vad_for_connection(self, connection_id: str):
        """
        Remove VAD instance for a connection.
        
        Args:
            connection_id: Unique identifier for the connection
        """
        if connection_id in self.vad_instances:
            # Flush any remaining speech before removing
            vad = self.vad_instances[connection_id]
            remaining_speech = vad.flush_remaining_speech()
            if remaining_speech:
                self.logger.info(f"Flushed remaining speech for connection {connection_id}: {len(remaining_speech)} bytes")
            
            del self.vad_instances[connection_id]
            self.stats['active_connections'] -= 1
            self.logger.info(f"Removed VAD instance for connection {connection_id}")
    
    def process_audio_chunk(self, connection_id: str, audio_data: bytes) -> Optional[bytes]:
        """
        Process audio chunk through VAD for a connection.
        
        Args:
            connection_id: Unique identifier for the connection
            audio_data: Raw audio data
            
        Returns:
            Complete speech segment if detected, None otherwise
        """
        if not self.enabled:
            return audio_data  # Return original audio if VAD is disabled
        
        vad = self.get_vad_for_connection(connection_id)
        if not vad:
            return audio_data  # Return original audio if no VAD instance
        
        try:
            # Process audio chunk through VAD
            speech_segment = vad.process_audio_chunk(audio_data)
            
            if speech_segment:
                self.stats['speech_segments_detected'] += 1
                self.logger.info(f"VAD detected speech segment for connection {connection_id}: {len(speech_segment)} bytes")
                return speech_segment
            else:
                self.stats['silence_chunks_filtered'] += 1
                if VAD_DEBUG_LOGGING:
                    self.logger.debug(f"VAD filtered silence for connection {connection_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error processing audio chunk for connection {connection_id}: {e}")
            return audio_data  # Return original audio on error
    
    def detect_barge_in(self, connection_id: str, audio_data: bytes) -> bool:
        """
        Detect barge-in (interruption) for a connection.
        
        Args:
            connection_id: Unique identifier for the connection
            audio_data: Raw audio data
            
        Returns:
            True if barge-in detected, False otherwise
        """
        if not self.enabled or not VAD_BARGE_IN_ENABLED:
            return False
        
        vad = self.get_vad_for_connection(connection_id)
        if not vad:
            return False
        
        try:
            is_speech = vad.is_speech_for_barge_in(audio_data)
            if is_speech:
                self.stats['barge_in_events'] += 1
                self.logger.info(f"VAD detected barge-in for connection {connection_id}")
            return is_speech
            
        except Exception as e:
            self.logger.error(f"Error detecting barge-in for connection {connection_id}: {e}")
            return False
    
    def is_speech(self, connection_id: str, audio_data: bytes) -> bool:
        """
        Check if audio contains speech for a connection.
        
        Args:
            connection_id: Unique identifier for the connection
            audio_data: Raw audio data
            
        Returns:
            True if speech detected, False otherwise
        """
        if not self.enabled:
            return True  # Assume speech if VAD is disabled
        
        vad = self.get_vad_for_connection(connection_id)
        if not vad:
            return True  # Assume speech if no VAD instance
        
        try:
            return vad.is_speech(audio_data)
        except Exception as e:
            self.logger.error(f"Error checking speech for connection {connection_id}: {e}")
            return True  # Assume speech on error
    
    def get_connection_stats(self, connection_id: str) -> Dict[str, Any]:
        """
        Get VAD statistics for a specific connection.
        
        Args:
            connection_id: Unique identifier for the connection
            
        Returns:
            Dictionary with VAD statistics
        """
        vad = self.get_vad_for_connection(connection_id)
        if not vad:
            return {}
        
        return vad.get_stats()
    
    def get_global_stats(self) -> Dict[str, Any]:
        """
        Get global VAD statistics.
        
        Returns:
            Dictionary with global VAD statistics
        """
        return {
            **self.stats,
            'enabled': self.enabled,
            'active_vad_instances': len(self.vad_instances),
            'barge_in_enabled': VAD_BARGE_IN_ENABLED
        }
    
    def shutdown(self):
        """Shutdown the VAD manager and clean up resources."""
        self.logger.info("Shutting down VAD manager...")
        
        # Flush all remaining speech and remove all instances
        for connection_id in list(self.vad_instances.keys()):
            self.remove_vad_for_connection(connection_id)
        
        self.logger.info("VAD manager shutdown complete")


# Global VAD manager instance
vad_manager = VADManager()

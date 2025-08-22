"""
Processors package for AI Interview Evaluation System.
Contains audio processing and evaluation logic.
"""

from .audio_processor import process_audio_file, create_evaluation_prompt

__all__ = ['process_audio_file', 'create_evaluation_prompt'] 
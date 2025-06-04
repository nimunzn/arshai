"""
Speech processing services for speech-to-text and text-to-speech.

This module provides clients for various speech service providers
to handle transcription (STT) and synthesis (TTS) operations.
"""

from arshai.speech.openai import OpenAISpeechClient
from arshai.speech.azure import AzureSpeechClient

__all__ = [
    "OpenAISpeechClient",
    "AzureSpeechClient",
    ""
] 
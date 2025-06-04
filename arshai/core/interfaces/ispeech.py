from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, Optional, List, BinaryIO, Union
from enum import Enum, StrEnum
from pydantic import BaseModel, Field

from .idto import IDTO


class STTFormat(StrEnum):
    """Supported formats for speech-to-text transcription."""
    TEXT = "text"
    JSON = "json"
    SRT = "srt"
    VTT = "vtt"


class TTSVoiceType(StrEnum):
    """Voice types for text-to-speech synthesis."""
    NEURAL = "neural"
    STANDARD = "standard"


class TTSFormat(StrEnum):
    """Audio output formats for text-to-speech synthesis."""
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    FLAC = "flac"


class ISpeechConfig(IDTO):
    """
    Configuration for speech services.
    
    This configuration focuses on structural settings that define
    the behavior of speech services rather than connection details.
    Sensitive information like API keys, endpoints, and secrets should
    be read from environment variables by the implementations.
    """
    # Core configuration (model selection)
    stt_model: str = Field(
        default="whisper-1", 
        description="Model to use for speech-to-text"
    )
    tts_model: Optional[str] = Field(
        default=None, 
        description="Model to use for text-to-speech"
    )
    tts_voice: Optional[str] = Field(
        default=None, 
        description="Voice to use for text-to-speech"
    )
    tts_voice_type: TTSVoiceType = Field(
        default=TTSVoiceType.NEURAL, 
        description="Type of voice to use"
    )
    
    # Provider-specific structural configuration (no sensitive data)
    provider: str = Field(
        default="openai", 
        description="Speech provider (e.g., 'openai', 'azure')"
    )
    region: Optional[str] = Field(
        default=None, 
        description="Region for the speech service (for providers that need it)"
    )


class ISTTInput(IDTO):
    """Input configuration for speech-to-text transcription."""
    audio_file: Union[str, BinaryIO] = Field(
        description="Path to audio file or file-like object to transcribe"
    )
    language: Optional[str] = Field(
        default=None, 
        description="Language code (e.g., 'en', 'fa') for transcription"
    )
    response_format: STTFormat = Field(
        default=STTFormat.TEXT, 
        description="Format of the transcription response"
    )
    prompt: Optional[str] = Field(
        default=None, 
        description="Optional prompt to guide the transcription"
    )
    temperature: float = Field(
        default=0.0, 
        description="Temperature for sampling, 0.0 means deterministic"
    )


class ISTTOutput(IDTO):
    """Output from speech-to-text transcription."""
    text: str = Field(
        description="The transcribed text"
    )
    language: Optional[str] = Field(
        default=None, 
        description="Detected or specified language of the audio"
    )
    segments: Optional[List[Dict[str, Any]]] = Field(
        default=None, 
        description="Segments information (timestamps, confidence, etc.)"
    )
    duration: Optional[float] = Field(
        default=None, 
        description="Duration of the audio in seconds"
    )


class ITTSInput(IDTO):
    """Input configuration for text-to-speech synthesis."""
    text: str = Field(
        description="Text to convert to speech"
    )
    voice: Optional[str] = Field(
        default=None, 
        description="Voice ID to use for speech synthesis"
    )
    output_format: TTSFormat = Field(
        default=TTSFormat.MP3, 
        description="Audio output format"
    )
    speed: float = Field(
        default=1.0, 
        description="Speed of speech, 1.0 is normal speed"
    )


class ITTSOutput(IDTO):
    """Output from text-to-speech synthesis."""
    audio_data: bytes = Field(
        description="The generated audio as bytes"
    )
    duration: Optional[float] = Field(
        default=None, 
        description="Duration of the generated audio in seconds"
    )
    format: TTSFormat = Field(
        description="Format of the generated audio"
    )


class ISpeechProcessor(Protocol):
    """Protocol for speech processing services.
    
    This protocol defines the standard interface for services that
    can convert speech to text (transcription) and text to speech (synthesis).
    
    Implementations should read sensitive connection information (API keys,
    endpoints, credentials) from environment variables rather than from the
    configuration object, which should only contain structural settings.
    """
    
    def __init__(self, config: ISpeechConfig) -> None:
        """Initialize the speech processor with configuration."""
        ...
    
    def transcribe(self, input: ISTTInput) -> ISTTOutput:
        """Transcribe speech to text.
        
        Args:
            input: Configuration for the transcription
            
        Returns:
            Transcription result
        """
        ...
    
    def synthesize(self, input: ITTSInput) -> ITTSOutput:
        """Synthesize text to speech.
        
        Args:
            input: Configuration for the speech synthesis
            
        Returns:
            Generated audio output
        """
        ... 
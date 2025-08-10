import os
import logging
import asyncio
from typing import Dict, Any, Optional, BinaryIO, Union, List
from pathlib import Path
import tempfile

from openai import OpenAI, AsyncOpenAI

try:
    import aiofiles
    import aiofiles.os
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False

from arshai.core.interfaces.ispeech import (
    ISpeechProcessor, 
    ISpeechConfig,
    ISTTInput, 
    ISTTOutput,
    ITTSInput, 
    ITTSOutput,
    STTFormat,
    TTSFormat
)

logger = logging.getLogger(__name__)

class OpenAISpeechClient(ISpeechProcessor):
    """OpenAI implementation of the speech processor interface.
    
    This client uses OpenAI's Whisper model for speech-to-text transcription
    and their TTS API for text-to-speech synthesis.
    """
    
    def __init__(self, config: ISpeechConfig):
        """Initialize the OpenAI speech client.
        
        Args:
            config: Configuration for the service (structural settings)
            
        Raises:
            ValueError: If the required environment variables are not set
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing OpenAI speech client with STT model: {self.config.stt_model}")
        
        # Initialize both sync and async clients
        self._client = self._initialize_client()
        self._async_client = self._initialize_async_client()
    
    def _initialize_client(self) -> Any:
        """Initialize the OpenAI client using environment variables.
        
        Returns:
            OpenAI client instance
        
        Raises:
            ValueError: If API key is not available in environment variables
        """
        # Check for API key in environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self.logger.error("OpenAI API key not found in environment variables")
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
            )
            
        return OpenAI(api_key=api_key)
    
    def _initialize_async_client(self) -> Any:
        """Initialize the async OpenAI client using environment variables.
        
        Returns:
            AsyncOpenAI client instance
        
        Raises:
            ValueError: If API key is not available in environment variables
        """
        # Check for API key in environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self.logger.error("OpenAI API key not found in environment variables")
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
            )
            
        return AsyncOpenAI(api_key=api_key)
    
    def transcribe(self, input: ISTTInput) -> ISTTOutput:
        """Transcribe speech to text using OpenAI's Whisper model.
        
        Args:
            input: Configuration for the transcription
            
        Returns:
            Transcription result
            
        Raises:
            Exception: If transcription fails
        """
        self.logger.info(f"Transcribing audio with model: {self.config.stt_model}")
        
        # Process file input
        file_to_transcribe = self._prepare_audio_file(input.audio_file)
        
        try:
            # Call the OpenAI transcription API
            response_format = str(input.response_format.value)
            
            response = self._client.audio.transcriptions.create(
                model=self.config.stt_model,
                file=file_to_transcribe,
                language=input.language,
                response_format=response_format,
                prompt=input.prompt,
                temperature=input.temperature
            )
            
            # Parse the response based on format
            if input.response_format == STTFormat.JSON:
                text = response.text
                segments = response.segments if hasattr(response, 'segments') else None
                duration = response.duration if hasattr(response, 'duration') else None
                language = response.language if hasattr(response, 'language') else input.language
            else:
                # For TEXT, SRT, VTT formats
                text = response if isinstance(response, str) else str(response)
                segments = None
                duration = None
                language = input.language
            
            return ISTTOutput(
                text=text,
                language=language,
                segments=segments,
                duration=duration
            )
            
        except Exception as e:
            self.logger.error(f"Error transcribing with OpenAI: {str(e)}")
            raise
    
    def synthesize(self, input: ITTSInput) -> ITTSOutput:
        """Synthesize text to speech using OpenAI's TTS API.
        
        Args:
            input: Configuration for the speech synthesis
            
        Returns:
            Generated audio output
            
        Raises:
            Exception: If synthesis fails
        """
        self.logger.info(f"Synthesizing speech with voice: {input.voice or self.config.tts_voice or 'alloy'}")
        
        try:
            # Set default voice if not provided
            voice = input.voice or self.config.tts_voice or "alloy"
            model = self.config.tts_model or "tts-1"
            
            # Call the OpenAI TTS API
            response = self._client.audio.speech.create(
                model=model,
                voice=voice,
                input=input.text,
                speed=input.speed,
                response_format=input.output_format.value
            )
            
            # Get audio content
            audio_data = response.content
            
            return ITTSOutput(
                audio_data=audio_data,
                format=input.output_format,
                duration=None  # OpenAI doesn't provide duration info
            )
            
        except Exception as e:
            self.logger.error(f"Error synthesizing speech with OpenAI: {str(e)}")
            raise
    
    def _prepare_audio_file(self, audio_file: Union[str, BinaryIO]) -> BinaryIO:
        """Prepare the audio file for API request.
        
        Args:
            audio_file: Path to audio file or file-like object
            
        Returns:
            File object ready for API request
            
        Raises:
            FileNotFoundError: If the audio file cannot be found
        """
        # If input is a string (file path), open the file
        if isinstance(audio_file, str):
            path = Path(audio_file)
            if not path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_file}")
                
            return open(path, "rb")
        
        # If input is already a file-like object, return it
        if hasattr(audio_file, 'seek'):
            audio_file.seek(0)
        return audio_file
    
    # ----------------------
    # Async Methods (New - Using aiofiles and AsyncOpenAI)
    # ----------------------
    
    async def transcribe_async(self, input: ISTTInput) -> ISTTOutput:
        """Asynchronously transcribe speech to text using OpenAI's Whisper model.
        
        Args:
            input: Configuration for the transcription
            
        Returns:
            Transcription result
            
        Raises:
            Exception: If transcription fails
        """
        self.logger.info(f"Async transcribing audio with model: {self.config.stt_model}")
        
        # Process file input asynchronously
        file_to_transcribe = await self._prepare_audio_file_async(input.audio_file)
        
        try:
            # Call the async OpenAI transcription API
            response_format = str(input.response_format.value)
            
            response = await self._async_client.audio.transcriptions.create(
                model=self.config.stt_model,
                file=file_to_transcribe,
                language=input.language,
                response_format=response_format,
                prompt=input.prompt,
                temperature=input.temperature
            )
            
            # Parse the response based on format
            if input.response_format == STTFormat.JSON:
                text = response.text
                segments = response.segments if hasattr(response, 'segments') else None
                duration = response.duration if hasattr(response, 'duration') else None
                language = response.language if hasattr(response, 'language') else input.language
            else:
                # For TEXT, SRT, VTT formats
                text = response if isinstance(response, str) else str(response)
                segments = None
                duration = None
                language = input.language
            
            return ISTTOutput(
                text=text,
                language=language,
                segments=segments,
                duration=duration
            )
            
        except Exception as e:
            self.logger.error(f"Error transcribing with OpenAI async: {str(e)}")
            raise
        finally:
            # Clean up file handle if we opened it
            if hasattr(file_to_transcribe, 'close') and isinstance(input.audio_file, str):
                file_to_transcribe.close()
    
    async def synthesize_async(self, input: ITTSInput) -> ITTSOutput:
        """Asynchronously synthesize text to speech using OpenAI's TTS API.
        
        Args:
            input: Configuration for the speech synthesis
            
        Returns:
            Generated audio output
            
        Raises:
            Exception: If synthesis fails
        """
        self.logger.info(f"Async synthesizing speech with voice: {input.voice or self.config.tts_voice or 'alloy'}")
        
        try:
            # Set default voice if not provided
            voice = input.voice or self.config.tts_voice or "alloy"
            model = self.config.tts_model or "tts-1"
            
            # Call the async OpenAI TTS API
            response = await self._async_client.audio.speech.create(
                model=model,
                voice=voice,
                input=input.text,
                speed=input.speed,
                response_format=input.output_format.value
            )
            
            # Get audio content
            audio_data = response.content
            
            return ITTSOutput(
                audio_data=audio_data,
                format=input.output_format,
                duration=None  # OpenAI doesn't provide duration info
            )
            
        except Exception as e:
            self.logger.error(f"Error synthesizing speech with OpenAI async: {str(e)}")
            raise
    
    async def _prepare_audio_file_async(self, audio_file: Union[str, BinaryIO]) -> BinaryIO:
        """Asynchronously prepare the audio file for API request using aiofiles.
        
        Args:
            audio_file: Path to audio file or file-like object
            
        Returns:
            File object ready for API request
            
        Raises:
            FileNotFoundError: If the audio file cannot be found
        """
        # If input is a string (file path), open the file asynchronously
        if isinstance(audio_file, str):
            path = Path(audio_file)
            
            # Check if file exists asynchronously
            if HAS_AIOFILES:
                try:
                    # Use aiofiles.os.path.exists equivalent
                    stat_info = await aiofiles.os.stat(str(path))
                    if not stat_info:
                        raise FileNotFoundError(f"Audio file not found: {audio_file}")
                except FileNotFoundError:
                    raise FileNotFoundError(f"Audio file not found: {audio_file}")
                
                # Open file asynchronously and read content
                async with aiofiles.open(path, "rb") as f:
                    content = await f.read()
                
                # Create a BytesIO object for the API
                from io import BytesIO
                file_obj = BytesIO(content)
                file_obj.name = path.name  # Set name attribute for API
                return file_obj
            else:
                # Fallback to sync method if aiofiles not available
                self.logger.warning("aiofiles not available, falling back to sync file operations")
                return await asyncio.to_thread(self._prepare_audio_file, audio_file)
        
        # If input is already a file-like object, return it
        if hasattr(audio_file, 'seek'):
            audio_file.seek(0)
        return audio_file 
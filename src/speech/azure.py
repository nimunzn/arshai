import os
import logging
import io
from typing import BinaryIO, Union
from openai import AzureOpenAI

from arshai.core.interfaces import (
    ISpeechConfig,
    ISpeechProcessor,
    ISTTInput,
    ISTTOutput, 
    ITTSInput,
    ITTSOutput,
    STTFormat,
    TTSFormat
)

logger = logging.getLogger(__name__)

class AzureSpeechClient(ISpeechProcessor):
    """Client for Azure Speech services using Azure OpenAI.
    
    This client uses Azure OpenAI for both transcription (STT) and synthesis (TTS).
    """
    
    def __init__(self, config: ISpeechConfig) -> None:
        """Initialize Azure Speech client with OpenAI.
        
        Args:
            config: Configuration for Azure Speech services (structural settings)
        
        Raises:
            ValueError: If required environment variables are missing
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing Azure speech client with region: {self.config.region}")
        
        # Initialize the client
        self.azure_openai_client = self._initialize_client()
    
    def _initialize_client(self) -> AzureOpenAI:
        """Initialize the Azure OpenAI client using environment variables.
        
        Returns:
            AzureOpenAI client instance
        
        Raises:
            ValueError: If required environment variables are missing
        """
        # Get necessary configuration from environment variables
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if not api_key:
            self.logger.error("Azure OpenAI API key not found in environment variables")
            raise ValueError(
                "Azure OpenAI API key not found. Please set AZURE_OPENAI_API_KEY environment variable."
            )
        
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            self.logger.error("Azure OpenAI endpoint not found in environment variables")
            raise ValueError(
                "Azure OpenAI endpoint not found. Please set AZURE_OPENAI_ENDPOINT environment variable."
            )
        
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15")
        
        # Initialize Azure OpenAI client
        return AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
    
    def transcribe(self, input: ISTTInput) -> ISTTOutput:
        """Transcribe audio using Azure OpenAI's Whisper model.
        
        Args:
            input: Configuration for the transcription
            
        Returns:
            Transcription result
            
        Raises:
            FileNotFoundError: If the audio file doesn't exist
            Exception: If transcription fails
        """
        try:
            # Get deployment name from environment variable or use the model name from config
            deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME") or self.config.stt_model
            self.logger.info(f"Transcribing audio with deployment: {deployment_name}")
            
            # Prepare the audio file
            audio_file = self._prepare_audio_file(input.audio_file)
            
            # Call Azure OpenAI API
            response = self.azure_openai_client.audio.transcriptions.create(
                file=audio_file,
                model=deployment_name,
                language=input.language,
                response_format=input.response_format.value,
                prompt=input.prompt,
                temperature=input.temperature
            )
            
            # Parse the response based on format
            if input.response_format == STTFormat.TEXT:
                return ISTTOutput(
                    text=response.text,
                    language=input.language
                )
            else:
                # For JSON/SRT/VTT, the response contains additional information
                result = ISTTOutput(
                    text=response.text,
                    language=input.language
                )
                
                # Add segments if available (in JSON format)
                if hasattr(response, "segments"):
                    result.segments = response.segments
                    
                # Add duration if available
                if hasattr(response, "duration"):
                    result.duration = response.duration
                
                return result
                
        except FileNotFoundError as e:
            logger.error(f"Audio file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error transcribing audio with Azure OpenAI: {e}")
            raise
    
    def synthesize(self, input: ITTSInput) -> ITTSOutput:
        """Synthesize text to speech using Azure OpenAI.
        
        Args:
            input: Configuration for the speech synthesis
            
        Returns:
            Generated audio output
            
        Raises:
            Exception: If synthesis fails
        """
        try:
            # Get deployment name from environment variable or use the model name from config
            deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME") or self.config.tts_model or self.config.stt_model
            
            self.logger.info(f"Synthesizing speech with deployment: {deployment_name}, voice: {input.voice or self.config.tts_voice or 'alloy'}")
            
            # Call Azure OpenAI API for TTS
            response = self.azure_openai_client.audio.speech.create(
                model=deployment_name,
                voice=input.voice or self.config.tts_voice or "alloy",  # Default voice
                input=input.text,
                speed=input.speed,
                response_format=input.output_format.value
            )
            
            # Get the audio data as bytes
            audio_data = response.content
            
            return ITTSOutput(
                audio_data=audio_data,
                format=input.output_format,
                duration=None  # Azure OpenAI doesn't provide duration information directly
            )
            
        except Exception as e:
            logger.error(f"Error synthesizing speech with Azure OpenAI: {e}")
            raise
    
    def _prepare_audio_file(self, audio_input: Union[str, BinaryIO]) -> BinaryIO:
        """Prepare audio file for API request.
        
        Args:
            audio_input: Path to audio file or file-like object
            
        Returns:
            File-like object that can be sent to the API
            
        Raises:
            FileNotFoundError: If the audio file doesn't exist
        """
        # If it's already a file-like object, return it
        if hasattr(audio_input, 'read'):
            if hasattr(audio_input, 'seek'):
                audio_input.seek(0)
            return audio_input
        
        # Otherwise, open the file
        try:
            return open(audio_input, 'rb')
        except FileNotFoundError:
            logger.error(f"Audio file not found: {audio_input}")
            raise 
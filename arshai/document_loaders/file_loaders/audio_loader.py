from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from .base_loader import BaseFileLoader
from arshai.document_loaders.config import AudioLoaderConfig
from arshai.core.interfaces.itext_splitter import ITextSplitter
from arshai.core.interfaces.illm import ILLMInput, LLMInputType
from arshai.config import Settings
from arshai.core.interfaces.ispeech import ISTTInput, STTFormat

logger = logging.getLogger(__name__)

class AudioLoader(BaseFileLoader):
    """Audio file loader that transcribes audio to text.
    
    This loader uses speech processing services (like OpenAI's Whisper) to transcribe audio files,
    and can optionally correct the transcription using a language model.
    """
    
    def __init__(self, config: AudioLoaderConfig, setting: Settings, text_splitter: Optional[ITextSplitter] = None):
        """Initialize the audio loader.
        
        Args:
            config: Configuration for the loader
            setting: Settings instance for accessing components
            text_splitter: Optional text splitter for chunking documents
        """
        super().__init__(config, text_splitter)
        self.config = config  # Override with the correct type
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing AudioLoader")
        self.setting = setting
        
        # Get speech model and LLM from settings
        self.speech_processor = self.setting.create_speech_model()
        if not self.speech_processor:
            self.logger.warning("No speech model configured in settings. Audio transcription may fail.")
        
        # Initialize LLM client for transcription correction if needed
        self.llm_client = self.setting.create_llm()
    
    def _extract_content(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract content from an audio file by transcribing it.
        
        This method implements step 1 of our approach:
        - Extract raw content (transcription) without any processing
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            List of dictionaries with 'content' and 'metadata' keys
        """
        try:
            self.logger.info(f"Transcribing audio file: {file_path}")
            
            # Get file type
            file_type = self.get_file_type(file_path)
            
            # Create speech input
            stt_input = ISTTInput(
                audio_file=str(file_path),
                language=self.config.language,
                response_format=STTFormat.TEXT
            )
            
            # Transcribe the audio file using speech processor
            transcription = self.speech_processor.transcribe(stt_input)
            
            transcript_text = transcription.text
            
            # Correct the transcription if needed
            if self.config.correction_model and self.config.correction_model.startswith("gpt"):
                transcript_text = self._correct_transcript(transcript_text)
            
            # Create metadata
            metadata = {
                "source": str(file_path),
                "file_type": file_type,
                "language": self.config.language,
                "model": self.config.model
            }
            
            # Return a single content item
            return [{
                'content': transcript_text,
                'metadata': metadata
            }]
            
        except Exception as e:
            self.logger.error(f"Error transcribing audio file {file_path}: {str(e)}")
            raise
    
    def _correct_transcript(self, transcript: str) -> str:
        """Correct a transcript using a language model.
        
        Args:
            transcript: The raw transcript
            
        Returns:
            Corrected transcript
        """
        try:
            # Create a system prompt for correction
            system_prompt = """
            You are an intelligent assistant for correcting transcribed text. Your task is to correct any spelling errors in the text.
            Ensure that the following words are correctly spelled if they appear. Only make necessary punctuation changes such as periods, commas, and capital letters, and stay true to the given text.
            """
            
            # Prepare input for the LLM client
            llm_input = ILLMInput(
                input_type=LLMInputType.CHAT_COMPLETION,
                system_prompt=system_prompt,
                user_message=transcript
            )
            
            # Send to the correction model
            response = self.llm_client.chat_completion(llm_input)
            
            # Extract the corrected text
            if isinstance(response, dict) and "llm_response" in response:
                return response["llm_response"]
            return transcript  # Return original if something went wrong
            
        except Exception as e:
            self.logger.error(f"Error correcting transcript: {str(e)}")
            return transcript  # Return original if correction fails 
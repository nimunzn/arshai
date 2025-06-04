"""Unit tests for Azure Speech client."""

import pytest
from unittest.mock import MagicMock, patch, mock_open
import io
import os

from arshai.core.interfaces import (
    ISpeechConfig, 
    ISTTInput, 
    ITTSInput, 
    STTFormat, 
    TTSFormat
)
from arshai.speech.azure import AzureSpeechClient


@pytest.fixture
def speech_config():
    """Create a basic speech configuration for Azure."""
    return ISpeechConfig(
        provider="azure",
        stt_model="whisper-1",
        tts_model="tts-1",
        tts_voice="alloy",
        region="eastus"
    )


@pytest.fixture
def mock_openai_client():
    """Create a mock Azure OpenAI client."""
    mock_client = MagicMock()
    
    # Mock audio transcriptions create method
    mock_transcriptions = MagicMock()
    mock_client.audio.transcriptions.create = mock_transcriptions
    
    # Mock audio speech create method
    mock_speech = MagicMock()
    mock_client.audio.speech.create = mock_speech
    
    return mock_client


@pytest.fixture
def azure_speech_client(mock_openai_client, speech_config):
    """Create an Azure Speech client with mocked OpenAI client."""
    with patch("src.speech.azure.AzureOpenAI", return_value=mock_openai_client):
        with patch.dict("os.environ", {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_API_VERSION": "2024-02-15"
        }):
            client = AzureSpeechClient(speech_config)
            return client


def test_initialization():
    """Test initialization with different parameters."""
    # Test with environment variables
    with patch("src.speech.azure.AzureOpenAI") as mock_azure:
        with patch.dict("os.environ", {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_API_VERSION": "2024-02-15"
        }):
            config = ISpeechConfig(
                provider="azure",
                stt_model="whisper-1",
                region="eastus"
            )
            client = AzureSpeechClient(config)
            
            # Verify client was initialized with correct parameters
            mock_azure.assert_called_once_with(
                api_key="test-key",
                api_version="2024-02-15",
                azure_endpoint="https://test.openai.azure.com"
            )
    
    # Test with missing API key
    with patch.dict("os.environ", {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com"
    }, clear=True):
        config = ISpeechConfig(
            provider="azure",
            stt_model="whisper-1"
        )
        with pytest.raises(ValueError, match="Azure OpenAI API key not found"):
            AzureSpeechClient(config)
    
    # Test with missing endpoint
    with patch.dict("os.environ", {
        "AZURE_OPENAI_API_KEY": "test-key"
    }, clear=True):
        config = ISpeechConfig(
            provider="azure",
            stt_model="whisper-1"
        )
        with pytest.raises(ValueError, match="Azure OpenAI endpoint not found"):
            AzureSpeechClient(config)


def test_transcribe_with_file_path(azure_speech_client, mock_openai_client):
    """Test transcribing audio using a file path."""
    # Set up the mock response
    mock_response = MagicMock()
    mock_response.text = "This is the transcribed text"
    mock_openai_client.audio.transcriptions.create.return_value = mock_response
    
    # Set up the input
    input = ISTTInput(
        audio_file="test_audio.mp3",
        language="en",
        response_format=STTFormat.TEXT
    )
    
    # Mock the file open operation
    with patch("builtins.open", mock_open(read_data=b"audio data")) as mock_file:
        result = azure_speech_client.transcribe(input)
        
        # Verify file was opened
        mock_file.assert_called_once_with("test_audio.mp3", "rb")
        
        # Verify API was called with correct parameters
        mock_openai_client.audio.transcriptions.create.assert_called_once()
        args = mock_openai_client.audio.transcriptions.create.call_args[1]
        assert args["model"] == "whisper-1"  # From config
        assert args["language"] == "en"
        assert args["response_format"] == "text"
        
        # Verify result
        assert result.text == "This is the transcribed text"
        assert result.language == "en"


def test_transcribe_with_file_object(azure_speech_client, mock_openai_client):
    """Test transcribing audio using a file-like object."""
    # Set up the mock response
    mock_response = MagicMock()
    mock_response.text = "This is the transcribed text"
    mock_response.segments = [{"segment": 1}, {"segment": 2}]
    mock_response.duration = 10.5
    mock_openai_client.audio.transcriptions.create.return_value = mock_response
    
    # Create a file-like object
    audio_data = io.BytesIO(b"audio data")
    
    # Patch the ISTTInput validation to accept BytesIO
    with patch("seedwork.interfaces.ispeech.ISTTInput", autospec=True) as mock_input_cls:
        mock_input = MagicMock()
        mock_input.audio_file = audio_data
        mock_input.language = "en"
        mock_input.response_format = STTFormat.JSON
        mock_input_cls.return_value = mock_input
        
        # Create the input directly
        input = mock_input
        
        result = azure_speech_client.transcribe(input)
        
        # Verify API was called with correct parameters
        mock_openai_client.audio.transcriptions.create.assert_called_once()
        args = mock_openai_client.audio.transcriptions.create.call_args[1]
        assert args["file"] == audio_data
        assert args["response_format"] == "json"
        
        # Verify result
        assert result.text == "This is the transcribed text"
        assert result.segments == [{"segment": 1}, {"segment": 2}]
        assert result.duration == 10.5
        assert result.language == "en"  # From input


def test_transcribe_error_handling(azure_speech_client, mock_openai_client):
    """Test error handling in transcription."""
    # Set up the mock to raise an exception
    mock_openai_client.audio.transcriptions.create.side_effect = Exception("API error")
    
    # Set up the input
    input = ISTTInput(
        audio_file="nonexistent_file.mp3",
        language="en"
    )
    
    # Test file not found error
    with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
        with pytest.raises(FileNotFoundError):
            azure_speech_client.transcribe(input)
    
    # Test API error
    with patch("builtins.open", mock_open(read_data=b"audio data")):
        with pytest.raises(Exception, match="API error"):
            azure_speech_client.transcribe(input)


def test_synthesize(azure_speech_client, mock_openai_client):
    """Test text-to-speech synthesis."""
    # Set up the mock response
    mock_response = MagicMock()
    mock_response.content = b"synthesized audio data"
    mock_openai_client.audio.speech.create.return_value = mock_response
    
    # Set up the input
    input = ITTSInput(
        text="This is text to convert to speech",
        voice="alloy",
        output_format=TTSFormat.MP3,
        speed=1.2
    )
    
    result = azure_speech_client.synthesize(input)
    
    # Verify API was called with correct parameters
    mock_openai_client.audio.speech.create.assert_called_once()
    args = mock_openai_client.audio.speech.create.call_args[1]
    assert args["model"] == "tts-1"  # From config
    assert args["voice"] == "alloy"
    assert args["input"] == "This is text to convert to speech"
    assert args["speed"] == 1.2
    assert args["response_format"] == "mp3"
    
    # Verify result
    assert result.audio_data == b"synthesized audio data"
    assert result.format == TTSFormat.MP3
    assert result.duration is None  # Azure doesn't provide duration


def test_synthesize_default_voice(azure_speech_client, mock_openai_client):
    """Test synthesis with default voice from config."""
    # Set up the mock response
    mock_response = MagicMock()
    mock_response.content = b"synthesized audio data"
    mock_openai_client.audio.speech.create.return_value = mock_response
    
    # Set up the input without specifying voice
    input = ITTSInput(
        text="This is text to convert to speech",
        output_format=TTSFormat.WAV
    )
    
    result = azure_speech_client.synthesize(input)
    
    # Verify API was called with voice from config
    args = mock_openai_client.audio.speech.create.call_args[1]
    assert args["voice"] == "alloy"  # From config
    assert args["response_format"] == "wav"
    
    # Verify result
    assert result.format == TTSFormat.WAV


def test_synthesize_error_handling(azure_speech_client, mock_openai_client):
    """Test error handling in synthesis."""
    # Set up the mock to raise an exception
    mock_openai_client.audio.speech.create.side_effect = Exception("API error")
    
    # Set up the input
    input = ITTSInput(
        text="This is text to convert to speech",
        output_format=TTSFormat.MP3
    )
    
    # Test API error
    with pytest.raises(Exception, match="API error"):
        azure_speech_client.synthesize(input)


def test_deployment_name_from_env(azure_speech_client, mock_openai_client):
    """Test using deployment name from environment variables."""
    # Set up the mock response
    mock_response = MagicMock()
    mock_response.text = "This is the transcribed text"
    mock_openai_client.audio.transcriptions.create.return_value = mock_response
    
    # Set up the input
    input = ISTTInput(
        audio_file="test_audio.mp3",
        language="en"
    )
    
    # Mock the file open operation and add deployment name env var
    with patch("builtins.open", mock_open(read_data=b"audio data")):
        with patch.dict("os.environ", {"AZURE_OPENAI_DEPLOYMENT_NAME": "custom-deployment"}):
            result = azure_speech_client.transcribe(input)
            
            # Verify API was called with deployment name from env var
            args = mock_openai_client.audio.transcriptions.create.call_args[1]
            assert args["model"] == "custom-deployment" 
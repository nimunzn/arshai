# Speech Processing Module

This module provides a unified interface for various speech processing services, including:

- Speech-to-Text (STT) transcription
- Text-to-Speech (TTS) synthesis

## Supported Providers

- **OpenAI**: Uses Whisper for STT and TTS API for synthesis
- **Azure**: Uses Azure Cognitive Services Speech API

## Getting Started

### Configuration

Speech configuration can be set in two ways:

1. **YAML Configuration (Recommended):** Add a speech section to your config.yaml

```yaml
speech:
  provider: "openai"  # or "azure"
  stt_model: "whisper-1"
  tts_model: "tts-1"
  tts_voice: "alloy"  # OpenAI voices: alloy, echo, fable, onyx, nova, shimmer
  region: "eastus"  # Only for Azure
```

2. **Programmatic Configuration:** Create a configuration for your speech client using ISpeechConfig:

```python
from seedwork.interfaces.ispeech import ISpeechConfig, TTSVoiceType

# For OpenAI
speech_config = ISpeechConfig(
    provider="openai",
    stt_model="whisper-1",
    tts_model="tts-1",
    tts_voice="alloy"  # OpenAI voices: alloy, echo, fable, onyx, nova, shimmer
)

# For Azure
speech_config = ISpeechConfig(
    provider="azure",
    region="eastus",  # Optional: Azure region
    stt_model="whisper-1",
    tts_voice="en-US-AriaNeural",
    tts_voice_type=TTSVoiceType.NEURAL
)
```

### Setting Environment Variables

Speech clients read sensitive information from environment variables:

**For OpenAI:**
```bash
export OPENAI_API_KEY=your-api-key
```

**For Azure:**
```bash
export AZURE_OPENAI_API_KEY=your-api-key
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
export AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name # Optional, defaults to model name from config
export AZURE_OPENAI_API_VERSION=2024-02-15 # Optional, defaults to "2024-02-15"
```

### Initializing a Client

There are three ways to initialize a speech client:

#### 1. Using SpeechFactory (Recommended)

```python
from src.speech import SpeechFactory
from seedwork.interfaces.ispeech import ISpeechConfig

# Create a configuration
config = ISpeechConfig(
    provider="openai",
    stt_model="whisper-1",
    tts_model="tts-1"
)

# Create a speech processor using the factory
speech_processor = SpeechFactory.create("openai", config)
```

#### 2. Using Settings (For Applications)

If you're using the Arshai framework with a proper settings configuration:

```python
from src.config.settings import Settings

# Initialize settings with your config.yaml
settings = Settings("path/to/config.yaml")

# Get the speech model (will use the provider specified in config.yaml)
speech_model = settings.create_speech_model()
```

#### 3. Direct Instantiation (For Testing)

```python
from src.speech import OpenAISpeechClient, AzureSpeechClient
from seedwork.interfaces.ispeech import ISpeechConfig

# Initialize OpenAI speech client
config = ISpeechConfig(stt_model="whisper-1", tts_model="tts-1")
openai_speech = OpenAISpeechClient(config)

# Initialize Azure speech client
config = ISpeechConfig(region="eastus")
azure_speech = AzureSpeechClient(config)
```

### Speech-to-Text (Transcription)

```python
from seedwork.interfaces.ispeech import ISTTInput, STTFormat

# Prepare input for transcription
stt_input = ISTTInput(
    audio_file="path/to/audio.mp3",  # Can be a file path or file-like object
    language="en",  # Optional language code
    response_format=STTFormat.TEXT,  # TEXT, JSON, SRT, VTT
    prompt="This is a meeting about project planning",  # Optional prompt to guide transcription
    temperature=0.0  # Lower values are more deterministic
)

# Transcribe with the speech processor
result = speech_processor.transcribe(stt_input)
print(f"Transcription: {result.text}")
```

### Text-to-Speech (Synthesis)

```python
from seedwork.interfaces.ispeech import ITTSInput, TTSFormat
import os

# Prepare input for speech synthesis
tts_input = ITTSInput(
    text="Hello, this is a test of text-to-speech synthesis.",
    voice="en-US-AriaNeural",  # Optional, overrides config
    output_format=TTSFormat.MP3,  # MP3, WAV, OGG, FLAC
    speed=1.0  # Speed of speech (1.0 is normal)
)

# Synthesize with the speech processor
audio = speech_processor.synthesize(tts_input)
with open("output.mp3", "wb") as f:
    f.write(audio.audio_data)
```

## Error Handling

Speech clients use consistent error handling. It's recommended to wrap API calls in try-except blocks:

```python
try:
    result = speech_processor.transcribe(stt_input)
except Exception as e:
    print(f"Transcription error: {str(e)}")
```

## Requirements

- For OpenAI: `openai` Python package
- For Azure: `openai` Python package with Azure OpenAI support 
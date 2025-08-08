# Arshai Observability System

A comprehensive, non-intrusive observability layer for the Arshai LLM framework. This system provides production-ready monitoring, metrics collection, and tracing for LLM interactions with automatic provider detection and token-level performance analysis.

## 🚀 Key Features

### Core Metrics (As Requested)
- ✅ **`llm_time_to_first_token_seconds`** - Time from request start to first token
- ✅ **`llm_time_to_last_token_seconds`** - Time from request start to last token  
- ✅ **`llm_duration_first_to_last_token_seconds`** - Duration from first token to last token
- ✅ **`llm_completion_tokens`** - Count of completion tokens generated

### Advanced Features
- **Non-Intrusive Design**: Zero side effects on LLM calls
- **Constructor-Based Integration**: Clean, direct integration via client constructors
- **Automatic Provider Detection**: Works with OpenAI, Azure, Anthropic, Google Gemini
- **YAML Configuration Support**: Configure via `config.yaml` as per Arshai patterns
- **Token Counting**: Accurate token counting from LLM responses
- **Streaming Support**: Token-level timing for streaming responses
- **OpenTelemetry Compatible**: Full OTLP export support

## 📁 Architecture

```
arshai/observability/
├── __init__.py                 # Main exports
├── config.py                   # Configuration support
├── core.py                     # ObservabilityManager
├── metrics.py                  # MetricsCollector with key metrics
├── decorators.py               # DEPRECATED - decorator approach
├── factory_integration.py     # DEPRECATED - factory approach
├── helpers.py                  # DEPRECATED - helper functions
└── README.md                   # This file
```

## 🔧 Installation

The observability system is included with Arshai but requires optional dependencies for full functionality:

```bash
# Install OpenTelemetry dependencies
pip install opentelemetry-api opentelemetry-sdk
pip install opentelemetry-exporter-otlp-proto-grpc
```

## ⚡ Quick Start

### Simple Setup (Recommended)

```python
from arshai.llms.openai import OpenAIClient
from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.observability import ObservabilityManager, ObservabilityConfig

# 1. Create LLM configuration
llm_config = ILLMConfig(
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000
)

# 2. Create observability configuration
obs_config = ObservabilityConfig(
    service_name="my-ai-app",
    track_token_timing=True,
    collect_metrics=True,
    log_prompts=False  # For privacy
)

# 3. Create observability manager
obs_manager = ObservabilityManager(obs_config)

# 4. Create client with observability
client = OpenAIClient(llm_config, observability_manager=obs_manager)

# 5. Use normally - observability is automatic!
response = await client.chat(ILLMInput(
    system_prompt="You are a helpful assistant.",
    user_message="Explain machine learning briefly."
))

# Metrics are automatically collected:
# ✅ llm_time_to_first_token_seconds
# ✅ llm_time_to_last_token_seconds  
# ✅ llm_duration_first_to_last_token_seconds
# ✅ llm_completion_tokens
```

### Multi-Provider Support

```python
from arshai.llms.azure import AzureClient
from arshai.llms.google_genai import GeminiClient
from arshai.llms.openrouter import OpenRouterClient

# Same observability manager works with all providers
obs_manager = ObservabilityManager(obs_config)

# OpenAI
openai_client = OpenAIClient(config, observability_manager=obs_manager)

# Azure (with required parameters)
azure_client = AzureClient(
    config, 
    observability_manager=obs_manager,
    azure_deployment="my-deployment",
    api_version="2024-02-01"
)

# Gemini
gemini_client = GeminiClient(config, observability_manager=obs_manager)

# OpenRouter  
openrouter_client = OpenRouterClient(config, observability_manager=obs_manager)

# All clients automatically collect the same metrics!
```

### Streaming Support

```python
# Streaming automatically includes token timing
async for chunk in client.stream(input_data):
    if chunk.get('llm_response'):
        print(chunk['llm_response'], end='', flush=True)
    
    # Final chunk contains usage metrics with timing data
    if chunk.get('usage'):
        usage = chunk['usage']
        print(f"\n📊 Tokens: {usage['total_tokens']}")
        print(f"⏱️  First token: {usage.get('time_to_first_token', 'N/A')}ms")
```

## 🛠️ Configuration

### YAML Configuration

Create `observability.yaml`:

```yaml
observability:
  service_name: "my-ai-application"
  track_token_timing: true
  collect_metrics: true
  log_prompts: false
  
  # OpenTelemetry settings
  otlp_endpoint: "http://localhost:4317"
  otlp_headers:
    api-key: "your-jaeger-key"
  
  # Advanced settings
  enable_tracing: true
  enable_metrics: true
  max_prompt_length: 1000
  max_response_length: 5000
```

Load from YAML:
```python
from arshai.observability import ObservabilityConfig

obs_config = ObservabilityConfig.from_yaml("observability.yaml")
obs_manager = ObservabilityManager(obs_config)
client = OpenAIClient(config, observability_manager=obs_manager)
```

### Environment Variables

```bash
export ARSHAI_SERVICE_NAME="my-ai-app"
export ARSHAI_OTLP_ENDPOINT="http://localhost:4317"
export ARSHAI_TRACK_TOKEN_TIMING="true"
export ARSHAI_COLLECT_METRICS="true"
```

```python
# Auto-loads from environment
obs_config = ObservabilityConfig.from_env()
```

## 📊 Metrics & Monitoring

### Key Metrics Collected

| Metric | Description | Unit |
|--------|-------------|------|
| `llm_time_to_first_token_seconds` | Latency from request to first token | seconds |
| `llm_time_to_last_token_seconds` | Total response generation time | seconds |  
| `llm_duration_first_to_last_token_seconds` | Token generation duration | seconds |
| `llm_completion_tokens` | Number of completion tokens | count |

### Labels Added
- `provider`: LLM provider (openai, azure, gemini, etc.)
- `model`: Model name (gpt-4, gpt-3.5-turbo, etc.)  
- `service_name`: Your application name

### Jaeger/OpenTelemetry Integration

```python
# Observability data is automatically exported to your OTLP endpoint
obs_config = ObservabilityConfig(
    service_name="my-app",
    otlp_endpoint="http://jaeger:4317",
    enable_tracing=True,
    enable_metrics=True
)
```

View in Jaeger UI at `http://localhost:16686`

## 🔄 Migration from Old Approaches

### From Factory-Based Approach

**❌ Old (Deprecated):**
```python
from src.factories.llm_factory import LLMFactory

client = LLMFactory.create_with_observability(
    provider="openai",
    config=llm_config,
    observability_config=obs_config
)
```

**✅ New (Recommended):**
```python
from arshai.llms.openai import OpenAIClient

obs_manager = ObservabilityManager(obs_config)
client = OpenAIClient(llm_config, observability_manager=obs_manager)
```

### From Helper Functions

**❌ Old (Deprecated):**
```python  
from arshai.observability import create_observable_openai_client

client = create_observable_openai_client(config, obs_config)
```

**✅ New (Recommended):**
```python
from arshai.llms.openai import OpenAIClient
from arshai.observability import ObservabilityManager

obs_manager = ObservabilityManager(obs_config)
client = OpenAIClient(config, observability_manager=obs_manager)
```

### Method Names

**❌ Old:**
```python
response = client.chat_completion(input_data)
async for chunk in client.stream_completion(input_data):
```

**✅ New:**
```python
response = await client.chat(input_data)
async for chunk in client.stream(input_data):
```

## 🎯 Benefits of Constructor Approach

1. **Cleaner Code**: Direct constructor usage, no decorators or factory wrappers
2. **Better IDE Support**: Full type hints and autocomplete
3. **Easier Testing**: Simple mocking and dependency injection
4. **Explicit Dependencies**: Clear what each client needs
5. **Less Magic**: No hidden behavior, everything is explicit

## 📈 Production Usage

### Docker Compose with Jaeger

```yaml
# docker-compose.yml
version: '3.8'
services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # Jaeger UI
      - "4317:4317"    # OTLP gRPC receiver
    environment:
      - COLLECTOR_OTLP_ENABLED=true
      
  my-app:
    build: .
    environment:
      - ARSHAI_OTLP_ENDPOINT=http://jaeger:4317
      - ARSHAI_SERVICE_NAME=my-ai-app
    depends_on:
      - jaeger
```

### Kubernetes ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: observability-config
data:
  observability.yaml: |
    observability:
      service_name: "ai-service"
      track_token_timing: true
      collect_metrics: true
      otlp_endpoint: "http://jaeger-collector:4317"
```

## 🧪 Testing

```python
# Easy to test - just mock the observability manager
from unittest.mock import Mock

mock_obs_manager = Mock()
client = OpenAIClient(config, observability_manager=mock_obs_manager)

# Or test without observability
client = OpenAIClient(config)  # No observability - works fine!
```

## 🔍 Troubleshooting

### Common Issues

1. **Missing Metrics**: Ensure OpenTelemetry dependencies are installed
2. **No Traces**: Check OTLP endpoint is reachable
3. **Performance Impact**: Observability has minimal overhead when properly configured

### Debug Mode

```python
obs_config = ObservabilityConfig(
    service_name="debug-app",
    log_level="DEBUG"  # Enable debug logging
)
```

## 📚 Advanced Usage

### Custom Metrics

```python  
# Add custom metrics alongside Arshai observability
from opentelemetry import metrics

meter = metrics.get_meter(__name__)
custom_counter = meter.create_counter("my_custom_metric")

# Use with Arshai observability
obs_manager = ObservabilityManager(obs_config)
client = OpenAIClient(config, observability_manager=obs_manager)

# Both systems work together
custom_counter.add(1, {"operation": "chat_request"})
response = await client.chat(input_data)
```

The constructor-based approach is simple, clean, and production-ready! 🚀
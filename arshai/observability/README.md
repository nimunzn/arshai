# Arshai Observability System

A comprehensive, non-intrusive observability layer for the Arshai LLM framework with Phoenix AI integration. This system provides production-ready monitoring, metrics collection, and tracing for LLM interactions with automatic input/output capture, provider detection, and token-level performance analysis.

## 🚀 Key Features

### Core Metrics (As Requested)
- ✅ **`llm_time_to_first_token_seconds`** - Time from request start to first token
- ✅ **`llm_time_to_last_token_seconds`** - Time from request start to last token  
- ✅ **`llm_duration_first_to_last_token_seconds`** - Duration from first token to last token
- ✅ **`llm_completion_tokens`** - Count of completion tokens generated

### Advanced Features
- **Phoenix AI Integration**: Advanced LLM interaction monitoring with comprehensive input/output tracing
- **Automatic Factory Integration**: Zero-code observability through intelligent factory wrapping
- **Real-time Input/Output Capture**: Automatic capture of prompts, responses, and usage metrics
- **Non-Intrusive Design**: Zero side effects on LLM calls with graceful degradation
- **Automatic Provider Detection**: Works with OpenAI, Azure, Anthropic, Google Gemini
- **YAML Configuration Support**: Configure via `config.yaml` as per Arshai patterns
- **Streaming Support**: Token-level timing for streaming responses with automatic capture
- **OpenTelemetry Compatible**: Full OTLP export support
- **Proper Span Naming**: Correct span names like `llm.chat_completion` instead of `llm.<lambda>`

## 📁 Architecture

```
arshai/observability/
├── __init__.py                 # Main exports
├── config.py                   # YAML configuration support with Phoenix
├── core.py                     # ObservabilityManager with Phoenix client
├── metrics.py                  # MetricsCollector with key metrics
├── factory_integration.py     # Automatic LLMFactory integration
├── decorators.py               # Auto-capture decorators
├── phoenix_client.py           # Phoenix AI platform integration
└── README.md                   # This file
```

## 🔧 Installation

The observability system is included with Arshai but requires optional dependencies for full functionality:

```bash
# Install OpenTelemetry dependencies
pip install opentelemetry-api opentelemetry-sdk
pip install opentelemetry-exporter-otlp-proto-grpc

# Install Phoenix AI observability
pip install arize-phoenix openinference-semantic-conventions

# Or install all observability features
pip install arshai[observability]
```

## ⚡ Quick Start

### 1. YAML Configuration (Recommended)

Create `config.yaml`:

```yaml
# config.yaml
llm:
  provider: azure  # or openai, anthropic, google
  model: gpt-4.1-mini
  temperature: 0.7

observability:
  # Service identification
  service_name: "my-arshai-app"
  environment: "production"
  
  # Core observability features
  trace_requests: true
  collect_metrics: true
  track_token_timing: true
  
  # Phoenix AI Platform integration
  phoenix_enabled: true
  phoenix_endpoint: "http://localhost:6006"
  
  # OpenTelemetry export
  otlp_endpoint: "http://localhost:4317"
  
  # Privacy controls for input/output capture
  log_prompts: true   # Enable for development
  log_responses: true # Enable for development
  max_prompt_length: 1000
  max_response_length: 1000
```

### 2. Zero-Configuration Usage (NEW!)

```python
from arshai.config.settings import Settings
from arshai.core.interfaces.illm import ILLMInput

# Settings automatically detects observability configuration
settings = Settings()  # Loads config.yaml automatically

# Create LLM - observability is automatically enabled if configured
llm = settings.create_llm()

# All calls are automatically instrumented with ZERO configuration required!
input_data = ILLMInput(
    system_prompt="You are a helpful assistant with expertise in AI.",
    user_message="What are the benefits of distributed tracing in microservices?"
)

response = llm.chat_completion(input_data)
print(response['llm_response'])

# 🎉 Automatic capture includes:
# ✅ Input messages (system prompt + user message)  
# ✅ Output response (complete LLM response)
# ✅ Usage metrics (prompt/completion/total tokens)
# ✅ Timing data (first token, last token, total duration)
# ✅ Invocation parameters (model, temperature, provider)
# ✅ Proper span naming (llm.chat_completion)
# ✅ Phoenix AI Platform integration
# ✅ OpenTelemetry export to Jaeger/Prometheus
```

### 3. Environment Variables

```bash
export ARSHAI_TRACK_TOKEN_TIMING=true
export ARSHAI_SERVICE_NAME=my-app
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

## 📊 Metrics Reference

### Request Metrics
- `llm_requests_total`: Total number of LLM requests
- `llm_requests_failed`: Total number of failed requests
- `llm_active_requests`: Number of currently active requests
- `llm_request_duration_seconds`: Total request duration histogram

### Token Metrics (Core Features)
- **`llm_time_to_first_token_seconds`**: Time from start to first token ⭐
- **`llm_time_to_last_token_seconds`**: Time from start to last token ⭐
- **`llm_duration_first_to_last_token_seconds`**: Duration between tokens ⭐
- **`llm_completion_tokens`**: Count of completion tokens ⭐
- `llm_prompt_tokens`: Count of prompt tokens
- `llm_tokens_total`: Total token count
- `llm_tokens_per_second`: Token generation throughput

### Span Attributes (OpenInference Compatible)
- `llm.provider`: LLM provider name
- `llm.model_name`: Model name (renamed from llm.model)
- `llm.system`: AI system identifier (openai, anthropic, etc.)
- `llm.time_to_first_token`: Time to first token (seconds)
- `llm.time_to_last_token`: Time to last token (seconds)
- `llm.duration_first_to_last_token`: Duration between tokens (seconds)
- `llm.token_count.prompt`: Prompt token count (renamed from llm.usage.prompt_tokens)
- `llm.token_count.completion`: Completion token count (renamed from llm.usage.completion_tokens)
- `llm.token_count.total`: Total token count (renamed from llm.usage.total_tokens)
- `input.value`: Input content
- `output.value`: Output content
- `input.mime_type`: Input format
- `output.mime_type`: Output format
- `llm.input_messages`: Input messages for chat APIs
- `llm.output_messages`: Output messages from LLM
- `llm.invocation_parameters`: Model parameters (temperature, max_tokens, etc.)
- `llm.function_call`: Function call details
- `llm.cost.prompt`: Cost for input tokens
- `llm.cost.completion`: Cost for output tokens
- `llm.cost.total`: Total cost

## 🏭 Factory Integration

### Automatic Observability

```python
from src.factories.llm_factory import LLMFactory

# Method 1: Direct creation with observability
client = LLMFactory.create_with_observability(
    provider="openai",
    config=llm_config,
    config_path="config.yaml"
)

# Method 2: Get observable factory
observable_factory = LLMFactory.get_observable_factory(
    config_path="config.yaml"
)
client = observable_factory.create("openai", llm_config)
```

### Provider Support

The factory automatically detects and supports:

- **OpenAI**: Uses usage data from API responses
- **Azure OpenAI**: Uses usage data from API responses
- **Anthropic Claude**: Uses usage data from API responses
- **Google Gemini**: Uses usage data from API responses
- **OpenRouter**: Uses usage data from API responses

## 🔧 Configuration Options

### Complete Configuration

```yaml
observability:
  # Basic controls
  trace_requests: true
  collect_metrics: true
  
  # Service identification
  service_name: "arshai-llm"
  service_version: "1.0.0"
  environment: "production"
  
  # Key feature: Token timing
  
  # Privacy controls
  log_prompts: false      # Never enable in production
  log_responses: false    # Never enable in production
  max_prompt_length: 1000
  max_response_length: 1000
  
  # OpenTelemetry export
  otlp_endpoint: "http://localhost:4317"
  otlp_headers:
    Authorization: "Bearer token"
  otlp_timeout: 10
  
  # Non-intrusive mode (recommended)
  non_intrusive: true
  
  # Provider-specific settings
  provider_configs:
    openai:
      track_token_timing: true
    anthropic:
      track_token_timing: true
    google:
      track_token_timing: false
  
  # Custom attributes
  custom_attributes:
    team: "ai-platform"
    component: "llm-client"
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ARSHAI_TRACK_TOKEN_TIMING` | Enable token timing | `true` |
| `ARSHAI_SERVICE_NAME` | Service name | `arshai-llm` |
| `ARSHAI_NON_INTRUSIVE` | Non-intrusive mode | `true` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint | None |
| `OTEL_SERVICE_NAME` | Service name (OTEL) | None |

## 🎯 Non-Intrusive Design

The observability system is designed to have **zero side effects** on LLM calls:

### No Modification of Original Behavior
```python
# Original LLM client behavior is preserved
response = client.chat_completion(input_data)
# Response format and content unchanged

# Observability data is collected separately
# No additional latency or API calls
```

### Wrapper-Based Architecture
```python
# Non-intrusive wrapper preserves original instance
class ObservableWrapper:
    def __init__(self, wrapped_instance, provider_name, obs_manager):
        self._wrapped = wrapped_instance  # Original instance
        # Observability is added as a layer
```

### Graceful Degradation
```python
# If observability fails, LLM calls continue normally
try:
    # Collect metrics
    with observability_manager.observe_llm_call(...) as timing:
        result = original_llm_method(...)
        return result
except Exception:
    # Observability error doesn't affect LLM call
    return original_llm_method(...)
```

## 🔄 Streaming Support

The system provides comprehensive streaming observability with usage data extraction:

### Automatic Streaming Observability
```python
async def example_streaming():
    async for chunk in client.stream_completion(input_data):
        # Each chunk is automatically timed
        # First token timing recorded automatically
        # Last token timing recorded automatically
        
        if chunk.get('usage'):
            # Final usage data includes timing metrics
            pass
```

### Manual Async Streaming Observability
```python
async def manual_streaming_example():
    obs_manager = ObservabilityManager(config)
    
    # Async context manager for better performance
    async with obs_manager.observe_streaming_llm_call("openai", "gpt-4", "streaming") as timing:
        async for chunk in client.stream_completion(input_data):
            # Process each chunk asynchronously for better performance
            usage_data = await obs_manager.process_streaming_chunk(chunk, timing)
            
            if chunk.get('llm_response'):
                content = chunk['llm_response']
                # Content is automatically counted and timed with async methods
            
            # Final chunk contains complete usage information
            if usage_data:
                print(f"Tokens: {usage_data['total_tokens']}")
                print(f"Time to first token: {timing.time_to_first_token}s")
                print(f"Generation duration: {timing.duration_first_to_last_token}s")
                break
```

### Concurrent Request Performance
```python
async def concurrent_llm_requests():
    """Example showing async performance benefits."""
    
    # Multiple concurrent requests with observability
    tasks = []
    for i in range(10):
        task = client.chat_completion(input_data)
        tasks.append(task)
    
    # All requests run concurrently with async observability
    results = await asyncio.gather(*tasks)
    
    # Each request is individually tracked with full metrics
    # Much faster than sequential requests!
```

### Streaming Usage Data

The system supports provider-specific streaming usage data extraction:

#### OpenAI/Azure Streaming
- **Usage extraction**: Automatically extracts final usage data from last chunk
- **Chunk format**: Handles OpenAI's `choices[0].delta.content` format
- **Real-time timing**: Tracks token timing without custom counting

#### Anthropic Streaming  
- **Event-based**: Extracts usage from `message_stop` events
- **Usage mapping**: Maps `input_tokens`/`output_tokens` to standard format
- **Stream timing**: Records timing for each content chunk

#### Google Gemini Streaming
- **Content-based**: Extracts usage data from streaming responses
- **Stream timing**: Real-time timing tracking for content chunks

#### OpenRouter Streaming
- **API-compatible**: Uses OpenAI-compatible streaming format
- **Usage data**: Extracts final usage from stream completion

### Streaming Metrics
- **Time to first token**: Measured when first content chunk arrives
- **Time to last token**: Measured when final content chunk arrives  
- **Duration**: Calculated between first and last token timestamps
- **Usage data**: Final token counts extracted from LLM response
- **Chunk-level timing**: Each chunk timestamp recorded for analysis

## 🛠 Advanced Usage

### Manual Observability Manager

```python
from arshai.observability import ObservabilityManager, ObservabilityConfig

# Create manager
config = ObservabilityConfig.from_yaml("config.yaml")
manager = ObservabilityManager(config)

# Manual timing
with manager.observe_llm_call("openai", "gpt-4", "custom_call") as timing:
    # Your LLM call here
    result = make_llm_call()
    
    # Manual token counting
    timing.update_token_counts(prompt_tokens=10, completion_tokens=20)
    timing.record_token()
    
    return result
```

### Pre-Call Token Counting

```python
# Count tokens before making API call
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
]

token_info = manager.pre_call_token_count("openai", "gpt-4", messages)
print(f"Estimated tokens: {token_info['prompt_tokens']}")
```

### Custom Decorators

```python
from arshai.observability.decorators import with_observability

@with_observability("custom_provider")
def my_llm_method(self, input_data):
    # Your custom LLM implementation
    return {"llm_response": "...", "usage": usage_data}
```

## 📈 Deployment

### Development Setup

```yaml
observability:
  log_prompts: true    # OK for development
  log_responses: true  # OK for development
  # No OTLP endpoint - uses console exporters
```

### Production Setup

```yaml
observability:
  service_name: "production-llm-service"
  environment: "production"
  
  # Privacy first
  log_prompts: false
  log_responses: false
  
  # Export to observability backend
  otlp_endpoint: "https://your-otel-collector.com"
  otlp_headers:
    Authorization: "Bearer your-production-token"
  
  # Performance tuning
  trace_sampling_rate: 0.1  # Sample 10% of traces
  metric_export_interval: 30
```

### Popular Backends

Works with any OpenTelemetry-compatible backend:

- **Jaeger**: Distributed tracing
- **Prometheus + Grafana**: Metrics and visualization
- **Datadog**: Full observability platform
- **New Relic**: APM and monitoring
- **OpenTelemetry Collector**: Data pipeline hub

## 🐛 Troubleshooting

### Common Issues

1. **No metrics appearing**
   - Check `collect_metrics=true` in config
   - Verify OTLP endpoint configuration
   - Ensure OpenTelemetry dependencies installed

2. **Token metrics not showing**
   - Check `track_token_timing=true` in config
   - Verify LLM responses include usage data (`prompt_tokens`, `completion_tokens`, `total_tokens`)
   - Ensure provider is supported (OpenAI, Azure, Anthropic, Google)

3. **High memory usage**
   - Reduce `max_span_attributes` in config
   - Lower `trace_sampling_rate`
   - Use `non_intrusive=true` mode

### Debug Logging

```python
import logging
logging.getLogger("arshai.observability").setLevel(logging.DEBUG)
```

### Validation

```python
from arshai.observability import ObservabilityManager, ObservabilityConfig

config = ObservabilityConfig.from_yaml("config.yaml")
manager = ObservabilityManager(config)

print(f"Observability enabled: {manager.is_enabled()}")
print(f"Token timing enabled: {manager.is_token_timing_enabled('openai')}")
```

## 📚 Examples

See `examples/observability_usage_example.py` for comprehensive examples including:

- YAML configuration loading
- Factory integration
- Streaming observability
- Manual timing collection
- Token counting examples
- Error handling patterns

## 🤝 Contributing

To extend the observability system:

1. **Add new providers**: Implement `BaseTokenCounter` for new LLM providers
2. **Add metrics**: Extend `MetricsCollector` with new measurements
3. **Add exporters**: Create new OpenTelemetry exporter integrations
4. **Add configuration**: Extend `ObservabilityConfig` with new options

## 📄 License

This observability system is part of the Arshai framework and follows the project's licensing terms.
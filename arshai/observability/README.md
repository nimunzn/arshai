# LLM-Friendly Observability for Arshai

A well-behaved OpenTelemetry observability system designed specifically for LLM packages and applications.

## 🎯 Key Features

- **✅ OTEL Compliant**: Uses proper `get_tracer("arshai", "version")` pattern
- **✅ Self-Managed**: Automatically sets up OTEL when parent app doesn't have it
- **✅ Parent Respectful**: Detects and uses existing OTEL setup when available
- **✅ Docker Ready**: Built-in support for containerized observability stacks
- **✅ Graceful Fallbacks**: Works with and without OTEL dependencies
- **✅ Package Isolated**: Uses `ARSHAI_TELEMETRY_*` environment variables
- **✅ LLM Optimized**: Designed specifically for LLM observability patterns

## 🚀 Quick Start

### Basic Usage

```python
from arshai.observability import get_llm_observability

# Get observability instance (auto-configures from environment)
observability = get_llm_observability()

# Observe an LLM call
async with observability.observe_llm_call(
    provider="openai",
    model="gpt-4",
    method_name="chat"
) as timing_data:
    # Your LLM call here
    response = await llm_client.chat(messages)
    
    # Record usage data
    await observability.record_usage_data(timing_data, {
        'input_tokens': response.usage.input_tokens,
        'output_tokens': response.usage.output_tokens,
        'total_tokens': response.usage.total_tokens
    })
```

### Custom Configuration with OTLP Export

```python
from arshai.observability import get_llm_observability, PackageObservabilityConfig, ObservabilityLevel, SpanKind

# Create custom configuration with direct OTLP endpoint
config = PackageObservabilityConfig(
    enabled=True,
    level=ObservabilityLevel.INFO,
    trace_llm_calls=True,
    collect_metrics=True,
    track_token_timing=True,
    span_kind=SpanKind.LLM,
    # Direct OTLP export configuration
    otlp_endpoint="http://phoenix:4317",  # Your collector endpoint
    # Package identification
    package_name="my-ai-app",
    package_version="1.0.0",
    # Privacy settings
    log_prompts=False,  # Privacy setting
    log_responses=False  # Privacy setting
)

observability = get_llm_observability(config)
```

### Production Example

```python
config = PackageObservabilityConfig(
    enabled=True,
    level=ObservabilityLevel.INFO,
    trace_llm_calls=True,
    collect_metrics=True,
    track_token_timing=True,
    span_kind=SpanKind.LLM,
    # Export to Phoenix/Docker
    otlp_endpoint="http://phoenix:4317",
    # PRIVACY CONTROLS - Production defaults
    log_prompts=False,     # Don't log user prompts 
    log_responses=False,   # Don't log LLM responses
    max_prompt_length=500,
    max_response_length=500,
    package_name="taloan-support-agent",
    package_version="1.0.0",
)
```

### Testing/Debug Configuration

```python
config = PackageObservabilityConfig(
    enabled=True,
    level=ObservabilityLevel.DEBUG,
    trace_llm_calls=True,
    collect_metrics=True,
    track_token_timing=True,
    span_kind=SpanKind.LLM,
    otlp_endpoint="http://phoenix:4317",
    # PRIVACY CONTROLS - Testing (normally should be False)
    log_prompts=True,      # Enable for debugging
    log_responses=True,    # Enable for debugging  
    max_prompt_length=20000,    # Increased for complete capture
    max_response_length=20000,  # Increased for complete capture
    package_name="taloan-support-agent",
    package_version="1.0.0",
)
```

## 🐳 Docker & OTEL Collector Setup

### Prerequisites

Before using Arshai's observability system, you need to set up an **OTEL Collector** to receive and export telemetry data. Arshai will automatically configure itself to send data to your collector.

### Option 1: Docker Compose with Phoenix

```yaml
# docker-compose.yml
version: '3.8'
services:
  # Phoenix observability platform
  phoenix:
    image: arizephoenix/phoenix:latest
    ports:
      - "6006:6006"    # Phoenix UI
      - "4317:4317"    # OTLP gRPC receiver
      - "4318:4318"    # OTLP HTTP receiver
    environment:
      - PHOENIX_GRPC_PORT=4317

  # Your application
  your-app:
    image: your-app:latest
    environment:
      - ARSHAI_OTLP_ENDPOINT=http://phoenix:4317
    depends_on:
      - phoenix
```

### Option 2: Docker Compose with OTEL Collector + Jaeger

```yaml
# docker-compose.yml
version: '3.8'
services:
  # OTEL Collector
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    command: ["--config=/etc/otelcol-contrib/otel-collector.yml"]
    volumes:
      - ./otel-collector.yml:/etc/otelcol-contrib/otel-collector.yml
    ports:
      - "4317:4317"    # OTLP gRPC receiver
      - "4318:4318"    # OTLP HTTP receiver
    depends_on:
      - jaeger

  # Jaeger for tracing
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # Jaeger UI
      - "14268:14268"  # Jaeger HTTP receiver

  # Your application  
  your-app:
    image: your-app:latest
    environment:
      - ARSHAI_OTLP_ENDPOINT=http://otel-collector:4317
    depends_on:
      - otel-collector
```

### Option 3: OTEL Collector Configuration

Create `otel-collector.yml`:

```yaml
# otel-collector.yml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:

exporters:
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true
  
  prometheus:
    endpoint: "0.0.0.0:8889"

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [jaeger]
    
    metrics:
      receivers: [otlp]
      processors: [batch] 
      exporters: [prometheus]
```

### Configuration

Set the OTLP endpoint to match your collector:

```bash
# Environment variable
export ARSHAI_OTLP_ENDPOINT="http://localhost:4317"

# Or in your application
import os
os.environ['ARSHAI_OTLP_ENDPOINT'] = "http://your-collector:4317"
```

### Verification

1. Start your observability stack:
   ```bash
   docker-compose up -d
   ```

2. Run your Arshai application - you should see:
   ```
   INFO - Set up OTEL tracing export to http://phoenix:4317
   INFO - Set up OTEL metrics export
   ```

3. Check your observability platform:
   - **Phoenix**: http://localhost:6006
   - **Jaeger**: http://localhost:16686
   - **Prometheus**: http://localhost:8889/metrics

## 🏗️ Architecture

### Core Components

- **`LLMObservability`**: Main interface for LLM observability
- **`TelemetryManager`**: Low-level OTEL abstraction layer
- **`PackageObservabilityConfig`**: Package-specific configuration
- **`TimingData`**: Container for timing measurements

### Design Principles

1. **Never Create Providers**: Uses existing OTEL providers from parent application
2. **Namespace Isolation**: All configuration uses `ARSHAI_*` prefixes
3. **Graceful Degradation**: No-op implementations when OTEL unavailable
4. **Privacy First**: Optional content logging with length limits
5. **LLM Focused**: Metrics designed specifically for LLM operations

## 📊 Metrics Collected

### Key LLM Metrics

- **Time to First Token**: `arshai_llm_time_to_first_token_seconds`
- **Time to Last Token**: `arshai_llm_time_to_last_token_seconds`
- **Request Duration**: `arshai_llm_request_duration_seconds`
- **Token Counts**: Input, output, and total tokens per request
- **Request Counters**: Success/failure rates by provider
- **Active Requests**: Current number of active LLM requests

### Span Attributes

- `llm.system`: Provider name (e.g., "openai", "anthropic")
- `llm.request.model`: Model name (e.g., "gpt-4", "claude-3")
- `llm.operation.name`: Method name (e.g., "chat", "stream")
- `llm.usage.input_tokens`: Number of input tokens
- `llm.usage.output_tokens`: Number of output tokens
- `arshai.component`: Component type (e.g., "llm_client", "agent")

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `ARSHAI_TELEMETRY_ENABLED` | Master switch for all telemetry | `true` | `true`/`false` |
| `ARSHAI_TELEMETRY_LEVEL` | Verbosity level | `INFO` | `OFF`/`ERROR`/`INFO`/`DEBUG` |
| `ARSHAI_OTLP_ENDPOINT` | **OTLP collector endpoint (required for export)** | `None` | `http://phoenix:4317` |
| `ARSHAI_SPAN_KIND` | OpenInference span kind | `LLM` | `LLM`/`AGENT`/`TOOL`/etc |
| `ARSHAI_TRACE_LLM_CALLS` | Enable LLM call tracing | `true` | `true`/`false` |
| `ARSHAI_TRACE_AGENT_OPERATIONS` | Enable agent operation tracing | `true` | `true`/`false` |
| `ARSHAI_TRACE_WORKFLOW_EXECUTION` | Enable workflow tracing | `true` | `true`/`false` |
| `ARSHAI_COLLECT_METRICS` | Enable metrics collection | `true` | `true`/`false` |
| `ARSHAI_TRACK_TOKEN_TIMING` | Track token-level timing | `true` | `true`/`false` |
| `ARSHAI_TRACK_COST_METRICS` | Track cost metrics | `false` | `true`/`false` |
| `ARSHAI_LOG_PROMPTS` | Log prompts (privacy sensitive) | `false` | `true`/`false` |
| `ARSHAI_LOG_RESPONSES` | Log responses (privacy sensitive) | `false` | `true`/`false` |

### Configuration from Environment

```python
from arshai.observability import PackageObservabilityConfig

# Load from environment variables
config = PackageObservabilityConfig.from_environment()
```

### Provider-Specific Configuration

```python
# Configure specific providers
config = config.configure_provider(
    provider="openai",
    enabled=True,
    track_token_timing=True,
    log_prompts=False
)
```

## 🔧 Integration Patterns

### LLM Client Integration

```python
from arshai.observability import get_llm_observability, PackageObservabilityConfig

class MyLLMClient:
    def __init__(self, config, observability_config=None):
        self.config = config
        self.observability = get_llm_observability(observability_config)
        self.provider_name = "my_provider"
    
    async def chat(self, messages):
        async with self.observability.observe_llm_call(
            provider=self.provider_name,
            model=self.config.model,
            method_name="chat"
        ) as timing_data:
            # Make LLM call
            response = await self._make_llm_call(messages)
            
            # Record usage
            await self.observability.record_usage_data(timing_data, {
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.total_tokens
            })
            
            return response
```

### Using Decorators

```python
from arshai.observability.utils import observe_llm_method

class LLMClient:
    @observe_llm_method("openai", "gpt-4")
    async def chat(self, messages):
        # Automatically observed
        return await self.client.chat(messages)
```

### Using Mixins

```python
from arshai.observability.utils import ObservabilityMixin

class MyLLMClient(ObservabilityMixin):
    def __init__(self, config):
        super().__init__()
        self._setup_observability("my_provider")
    
    async def chat(self, messages):
        async with self._observe_llm_call("chat", self.model) as timing_data:
            response = await self._actual_chat(messages)
            await self._record_usage(timing_data, response.usage)
            return response
```

## 🏢 Integration Modes

Arshai's observability system supports three integration modes:

### Mode 1: Self-Managed (Recommended for Docker)

Set up your OTEL Collector (see Docker section above), then configure directly via config:

```python
# Direct configuration - recommended approach
from arshai.observability import get_llm_observability, PackageObservabilityConfig

config = PackageObservabilityConfig(
    enabled=True,
    otlp_endpoint="http://otel-collector:4317",  # Your collector endpoint
    package_name="my-app",
    package_version="1.0.0"
)

# Arshai automatically sets up OTEL and connects to your collector
observability = get_llm_observability(config)  # Auto-configures and exports
```

Alternative - via environment variable:

```python
import os
os.environ['ARSHAI_OTLP_ENDPOINT'] = "http://otel-collector:4317"

# Arshai automatically sets up OTEL and connects to your collector
from arshai.observability import get_llm_observability
observability = get_llm_observability()  # Auto-configures and exports
```

### Mode 2: Parent Application Setup

If your parent application already has OTEL configured, Arshai will detect and use it:

```python
# Parent application sets up OTEL
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Set up OTEL for your application
resource = Resource.create({"service.name": "my-ai-app"})
tracer_provider = TracerProvider(resource=resource)
trace.set_tracer_provider(tracer_provider)

# Configure exporters as needed for your application
otlp_exporter = OTLPSpanExporter(endpoint="http://your-collector:4317")
tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

# Arshai automatically detects and uses your OTEL setup
from arshai.observability import get_llm_observability
observability = get_llm_observability()  # Uses your existing setup
```

### Mode 3: Local Development (No Export)

For local development without observability infrastructure:

```python
# No setup needed - Arshai creates local-only providers
from arshai.observability import get_llm_observability
observability = get_llm_observability()  # Creates spans/metrics locally (no export)
```

## 📈 Observability Levels

### `OFF`
- No telemetry collected
- Completely disabled

### `ERROR`  
- Only error-level telemetry
- Failed requests and exceptions

### `INFO` (Default)
- Standard telemetry level  
- Request/response timing and metrics
- Success/failure rates

### `DEBUG`
- Verbose telemetry
- Detailed span attributes
- Content logging (if enabled)

## 🔐 Privacy and Security

### Content Logging Controls

```python
config = PackageObservabilityConfig(
    log_prompts=False,  # Don't log user prompts
    log_responses=False,  # Don't log LLM responses
    max_prompt_length=500,  # Limit logged content length
    max_response_length=500,
    track_cost_metrics=False  # Don't track cost data
)
```

### Safe Defaults

- **Prompts/Responses**: Not logged by default
- **Content Length**: Limited to 500 characters when enabled
- **Cost Metrics**: Disabled by default
- **Provider Isolation**: Each provider can be configured independently

## 🧪 Testing

### Without OTEL Dependencies

```python
# Works even without opentelemetry installed
from arshai.observability import get_llm_observability

observability = get_llm_observability()  # Uses no-op implementations
```

### Disabling for Tests

```python
from arshai.observability import disable_observability

# Completely disable observability
config = disable_observability()
observability = get_llm_observability(config)
```

### Environment-Based Testing

```bash
# Disable for tests
export ARSHAI_TELEMETRY_ENABLED=false

# Or set to minimal level
export ARSHAI_TELEMETRY_LEVEL=OFF
```

## 📚 Examples

See [`examples/llm_friendly_observability_example.py`](../../examples/llm_friendly_observability_example.py) for a complete working example showing:

- Parent OTEL setup
- Arshai observability configuration  
- LLM client integration
- Agent observability
- Environment configuration
- Graceful fallbacks

## 🚀 Simple Integration Examples

These examples show how to integrate Arshai observability into your applications using both deployment modes.

### Example 1: Self-Managed Mode (Docker/Containerized Apps)

**Perfect for:** New applications, Docker deployments, microservices

```python
#!/usr/bin/env python3
"""
Simple example: Self-managed observability mode
Arshai automatically sets up OTEL and exports to your collector
"""

import asyncio
from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.llms.openai import OpenAIClient
from arshai.observability import PackageObservabilityConfig, SpanKind

async def main():
    # Step 1: Configure self-managed observability
    obs_config = PackageObservabilityConfig(
        enabled=True,
        
        # Self-managed mode: Provide OTLP endpoint 
        otlp_endpoint="http://localhost:4320",  # Your OTEL Collector
        
        # Package identification
        package_name="my-ai-app",
        package_version="1.0.0",
        
        # LLM-specific settings
        trace_llm_calls=True,
        collect_metrics=True,
        track_token_timing=True,
        span_kind=SpanKind.LLM,
        
        # Privacy settings (for production)
        log_prompts=False,  # Don't log user prompts
        log_responses=False  # Don't log LLM responses
    )
    
    # Step 2: Create LLM client with observability
    llm_config = ILLMConfig(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=100
    )
    
    # Arshai automatically sets up OTEL and exports to your collector
    client = OpenAIClient(llm_config, obs_config)
    
    # Step 3: Use the client - observability is automatic
    llm_input = ILLMInput(
        system_prompt="You are a helpful assistant",
        user_message="What is the capital of France?"
    )
    
    print("🚀 Making LLM call with self-managed observability...")
    response = await client.chat(llm_input)
    print(f"📝 Response: {response['llm_response']}")
    print(f"📊 Usage: {response['usage']}")
    
    print("\n✅ Traces and metrics automatically exported to: http://localhost:4320")
    print("📊 View in Phoenix: http://localhost:6006")

if __name__ == "__main__":
    asyncio.run(main())
```

**Environment Alternative:**
```python
import os
os.environ['ARSHAI_OTLP_ENDPOINT'] = 'http://localhost:4320'

# Then use default config - it will auto-configure
from arshai.observability import get_llm_observability
observability = get_llm_observability()  # Auto-detects endpoint from env
```

### Example 2: Parent Application Mode (Existing OTEL Apps)

**Perfect for:** Applications that already use OpenTelemetry, enterprise applications

```python
#!/usr/bin/env python3
"""
Simple example: Parent application mode
Your app sets up OTEL, Arshai automatically detects and uses it
"""

import asyncio
from arshai.core.interfaces.illm import ILLMConfig, ILLMInput  
from arshai.llms.openai import OpenAIClient
from arshai.observability import PackageObservabilityConfig, SpanKind

def setup_parent_otel():
    """Set up your application's OTEL configuration"""
    print("🏗️  Setting up parent application OTEL...")
    
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    
    # Your application's resource
    resource = Resource.create({
        "service.name": "my-enterprise-app",
        "service.version": "2.0.0",
        "deployment.environment": "production"
    })
    
    # Set up tracing
    tracer_provider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(endpoint="http://localhost:4320", insecure=True)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)
    
    # Set up metrics
    metric_exporter = OTLPMetricExporter(endpoint="http://localhost:4320", insecure=True)
    metric_reader = PeriodicExportingMetricReader(exporter=metric_exporter)
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)
    
    print("✅ Parent OTEL configuration complete")

async def main():
    # Step 1: Set up your application's OTEL configuration
    setup_parent_otel()
    
    # Step 2: Configure Arshai observability (NO otlp_endpoint needed)
    obs_config = PackageObservabilityConfig(
        enabled=True,
        
        # Parent application mode: NO otlp_endpoint
        # Arshai will detect and use your existing OTEL setup
        
        # Package identification (will be combined with parent resource)
        package_name="arshai",
        package_version="1.2.3",
        
        # LLM-specific settings
        trace_llm_calls=True,
        collect_metrics=True, 
        track_token_timing=True,
        span_kind=SpanKind.LLM,
        
        # Privacy settings
        log_prompts=False,
        log_responses=False
    )
    
    # Step 3: Create LLM client - it will use parent OTEL configuration
    llm_config = ILLMConfig(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=100
    )
    
    # Arshai automatically detects and uses your existing OTEL setup
    client = OpenAIClient(llm_config, obs_config)
    
    # Step 4: Use the client - spans integrate with your app's traces
    llm_input = ILLMInput(
        system_prompt="You are a helpful assistant",
        user_message="What is 2+2?"
    )
    
    print("🚀 Making LLM call with parent application OTEL...")
    response = await client.chat(llm_input)
    print(f"📝 Response: {response['llm_response']}")
    print(f"📊 Usage: {response['usage']}")
    
    print("\n✅ Traces integrated with your application's observability")
    print("🔗 Combined traces include both your app and Arshai spans")

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 3: Quick Start with Environment Variables

**Perfect for:** Quick testing, environment-based configuration

```python
#!/usr/bin/env python3
"""
Quick start example using environment variables
"""

import os
import asyncio

# Option A: Self-managed mode
os.environ['ARSHAI_OTLP_ENDPOINT'] = 'http://localhost:4320'
os.environ['ARSHAI_LOG_PROMPTS'] = 'false'  # Privacy
os.environ['ARSHAI_LOG_RESPONSES'] = 'false'  # Privacy

# Option B: Disable for testing
# os.environ['ARSHAI_TELEMETRY_ENABLED'] = 'false'

from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.llms.openai import OpenAIClient

async def main():
    # No observability config needed - reads from environment
    llm_config = ILLMConfig(model="gpt-4o-mini", temperature=0.0)
    client = OpenAIClient(llm_config)  # Auto-configured from environment
    
    llm_input = ILLMInput(
        system_prompt="You are a helpful assistant",
        user_message="Hello! How are you?"
    )
    
    print("🚀 Making LLM call with environment-configured observability...")
    response = await client.chat(llm_input)
    print(f"📝 Response: {response['llm_response']}")
    
    print("✅ Observability configured automatically from environment variables")

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 4: Production-Ready Configuration

**Perfect for:** Production deployments with full configuration

```python
#!/usr/bin/env python3
"""
Production-ready observability configuration
"""

import asyncio
from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.llms.openai import OpenAIClient
from arshai.observability import PackageObservabilityConfig, ObservabilityLevel, SpanKind

async def main():
    # Production configuration
    obs_config = PackageObservabilityConfig(
        enabled=True,
        level=ObservabilityLevel.INFO,  # Standard level for production
        
        # Self-managed export to Phoenix
        otlp_endpoint="http://phoenix:4317",
        
        # Service identification
        package_name="my-production-app",
        package_version="2.1.0",
        
        # LLM observability features
        trace_llm_calls=True,
        collect_metrics=True,
        track_token_timing=True,
        span_kind=SpanKind.LLM,
        
        # PRODUCTION PRIVACY SETTINGS - Very Important!
        log_prompts=False,        # Never log user prompts in production
        log_responses=False,      # Never log LLM responses in production
        max_prompt_length=100,    # Limit if accidentally enabled
        max_response_length=100,  # Limit if accidentally enabled
        track_cost_metrics=False, # Cost data may be sensitive
        
        # Custom attributes for filtering in observability platform
        custom_attributes={
            "environment": "production",
            "region": "us-east-1",
            "application": "customer-support-bot"
        }
    )
    
    # LLM configuration
    llm_config = ILLMConfig(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=500
    )
    
    # Create client
    client = OpenAIClient(llm_config, obs_config)
    
    # Use client
    llm_input = ILLMInput(
        system_prompt="You are a professional customer support assistant",
        user_message="I need help with my account"
    )
    
    print("🚀 Production LLM call with comprehensive observability...")
    response = await client.chat(llm_input)
    print(f"📝 Response: {response['llm_response'][:100]}...")
    print(f"📊 Usage: {response['usage']}")
    
    print("\n✅ Production observability active:")
    print("   🔒 Privacy: User content NOT logged")
    print("   📊 Metrics: Performance and usage tracked")
    print("   🔍 Traces: Request flows monitored")
    print("   📈 View in Phoenix: http://phoenix:6006")

if __name__ == "__main__":
    asyncio.run(main())
```

### Docker Compose Setup for Examples

To run these examples, set up observability infrastructure:

```yaml
# docker-compose.yml
version: '3.8'
services:
  # Phoenix for observability
  phoenix:
    image: arizephoenix/phoenix:latest
    ports:
      - "6006:6006"    # Phoenix UI
      - "4320:4317"    # OTLP gRPC (mapped to standard port)
      - "4321:4318"    # OTLP HTTP
    environment:
      - PHOENIX_GRPC_PORT=4317

  # Your application
  your-app:
    build: .
    environment:
      - ARSHAI_OTLP_ENDPOINT=http://phoenix:4317
      - OPENAI_API_KEY=your-api-key
    depends_on:
      - phoenix
```

### Quick Testing Commands

```bash
# Start observability infrastructure
docker-compose up -d phoenix

# Run examples
python example_self_managed.py      # Self-managed mode
python example_parent_otel.py       # Parent application mode
python example_environment.py       # Environment configuration
python example_production.py        # Production configuration

# View results
open http://localhost:6006  # Phoenix UI
```

## 🔄 Migration Guide

See [`MIGRATION_GUIDE.md`](./MIGRATION_GUIDE.md) for detailed migration instructions from the old observability system.

## 🤝 Best Practices

### DO ✅

- Use `get_llm_observability()` to get observability instance
- Configure via environment variables for flexibility
- Use package-specific `ARSHAI_*` environment variables
- Test with observability disabled
- Respect privacy settings for content logging

### DON'T ❌

- Create TracerProvider or MeterProvider instances
- Call `set_tracer_provider()` or `set_meter_provider()`
- Override parent application's OTEL configuration
- Log sensitive content by default
- Hard-code observability settings

## 📊 Integration with Observability Platforms

### Jaeger
```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:14268/api/traces
```

### Prometheus + Grafana
```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
# Configure collector to export to Prometheus
```

### Arize Phoenix
```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://app.phoenix.arize.com/v1/traces
```

### Elastic APM
```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://your-elastic-apm:8200
```

The LLM-friendly observability system works with any OTEL-compatible platform!

## 🆘 Troubleshooting

### Spans Not Appearing in Phoenix/Jaeger

1. **Check OTLP endpoint configuration:**
   ```bash
   export ARSHAI_OTLP_ENDPOINT="http://phoenix:4317"
   ```

2. **Verify your collector is running:**
   ```bash
   docker ps | grep phoenix  # or jaeger/otel-collector
   ```

3. **Check Arshai logs for connection:**
   ```
   INFO - Set up OTEL tracing export to http://phoenix:4317  ✅ Good
   WARNING - Failed to connect to any OTLP endpoints  ❌ Bad
   ```

4. **Ensure tracing is enabled:**
   ```bash
   export ARSHAI_TRACE_LLM_CALLS=true
   ```

### Metrics Not Exported

1. **Same OTLP endpoint check as above**

2. **Verify metrics collection is enabled:**
   ```bash
   export ARSHAI_COLLECT_METRICS=true
   ```

3. **Check your collector configuration supports metrics**

### No Observability Data At All

1. **Check if running in NoOp mode:**
   ```python
   from arshai.observability import get_llm_observability
   obs = get_llm_observability()
   print(f"Enabled: {obs.is_enabled()}")  # Should be True
   ```

2. **Verify OTEL packages are installed:**
   ```bash
   pip install opentelemetry-api opentelemetry-sdk
   pip install opentelemetry-exporter-otlp-proto-grpc
   ```

### High Memory Usage
- Reduce observability level: `ARSHAI_TELEMETRY_LEVEL=ERROR`
- Disable content logging: `ARSHAI_LOG_PROMPTS=false`
- Limit provider tracking to essential ones only

### Performance Impact
- Observability adds <1ms overhead when enabled
- No performance impact when disabled
- Use `ARSHAI_TELEMETRY_ENABLED=false` for maximum performance

## 📄 License

This observability system is part of the Arshai framework and follows the same MIT license.
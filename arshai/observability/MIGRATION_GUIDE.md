# LLM-Friendly Observability Migration Guide

This guide helps you migrate from the old observability system to the new LLM-friendly implementation that properly follows OpenTelemetry best practices.

## What Changed?

### ❌ Old System (Anti-Patterns)
- Created `TracerProvider` and `MeterProvider` instances
- Called `set_tracer_provider()` and `set_meter_provider()` 
- Configured exporters and processors directly
- Could override parent application's OTEL setup
- Hardcoded service configuration

### ✅ New System (LLM-Friendly)
- Uses `get_tracer("arshai", version)` pattern
- Never creates or sets OTEL providers
- Respects parent application's OTEL configuration
- Works with and without OTEL dependencies
- Package-specific environment variables

## Migration Steps

### 1. Update Imports

**Old:**
```python
from arshai.observability import ObservabilityManager, ObservabilityConfig

manager = ObservabilityManager(ObservabilityConfig())
```

**New:**
```python
from arshai.observability import get_llm_observability, PackageObservabilityConfig

observability = get_llm_observability(PackageObservabilityConfig.from_environment())
```

### 2. Update LLM Client Integration

**Old:**
```python
class MyLLMClient:
    def __init__(self, config, observability_manager=None):
        self.observability_manager = observability_manager
    
    async def chat(self, input):
        if self.observability_manager:
            async with self.observability_manager.observe_llm_call(...) as timing_data:
                # LLM call
                pass
```

**New:**
```python
from arshai.observability import get_llm_observability

class MyLLMClient:
    def __init__(self, config, observability_config=None):
        self.observability = get_llm_observability(observability_config)
        self.provider_name = "my_provider"
    
    async def chat(self, input):
        async with self.observability.observe_llm_call(
            provider=self.provider_name,
            model=self.config.model,
            method_name="chat"
        ) as timing_data:
            # LLM call
            response = await self._make_llm_call(input)
            
            # Record usage
            await self.observability.record_usage_data(timing_data, {
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.total_tokens
            })
            
            return response
```

### 3. Update Configuration

**Old Environment Variables:**
```bash
OTEL_SERVICE_NAME=my-app
OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4317
ARSHAI_TRACE_REQUESTS=true
```

**New Environment Variables:**
```bash
# Parent app still controls OTEL setup
OTEL_SERVICE_NAME=my-app
OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4317

# Package-specific controls
ARSHAI_TELEMETRY_ENABLED=true
ARSHAI_TELEMETRY_LEVEL=INFO
ARSHAI_TRACE_LLM_CALLS=true
ARSHAI_COLLECT_METRICS=true
ARSHAI_TRACK_TOKEN_TIMING=true
```

### 4. Update Usage Pattern

**Old:**
```python
# In your main application
from arshai.observability import ObservabilityManager, ObservabilityConfig

config = ObservabilityConfig(
    service_name="my-app",
    otlp_endpoint="http://collector:4317"
)
manager = ObservabilityManager(config)

# Use with LLM client
llm_client = MyLLMClient(llm_config, observability_manager=manager)
```

**New:**
```python
# In your main application - set up OTEL as usual
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Your app controls OTEL setup
tracer_provider = TracerProvider()
trace.set_tracer_provider(tracer_provider)
otlp_exporter = OTLPSpanExporter(endpoint="http://collector:4317")
# ... configure as needed

# Arshai automatically detects and uses your OTEL setup
from arshai.observability import PackageObservabilityConfig

# Optional: customize Arshai-specific settings
arshai_config = PackageObservabilityConfig.from_environment()
llm_client = MyLLMClient(llm_config, observability_config=arshai_config)
```

## Environment Variable Reference

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `ARSHAI_TELEMETRY_ENABLED` | Master switch for all telemetry | `true` | `true`/`false` |
| `ARSHAI_TELEMETRY_LEVEL` | Verbosity level | `INFO` | `OFF`/`ERROR`/`INFO`/`DEBUG` |
| `ARSHAI_TRACE_LLM_CALLS` | Enable LLM call tracing | `true` | `true`/`false` |
| `ARSHAI_TRACE_AGENT_OPERATIONS` | Enable agent operation tracing | `true` | `true`/`false` |
| `ARSHAI_TRACE_WORKFLOW_EXECUTION` | Enable workflow tracing | `true` | `true`/`false` |
| `ARSHAI_COLLECT_METRICS` | Enable metrics collection | `true` | `true`/`false` |
| `ARSHAI_TRACK_TOKEN_TIMING` | Track token-level timing | `true` | `true`/`false` |
| `ARSHAI_TRACK_COST_METRICS` | Track cost metrics | `false` | `true`/`false` |
| `ARSHAI_LOG_PROMPTS` | Log prompts (privacy sensitive) | `false` | `true`/`false` |
| `ARSHAI_LOG_RESPONSES` | Log responses (privacy sensitive) | `false` | `true`/`false` |

## Installation Options

### Core Package (No Observability)
```bash
pip install arshai
```

### With Observability Support
```bash
pip install arshai[observability]
```

### Full Installation
```bash
pip install arshai[full]  # Includes observability + other extras
```

## Compatibility

### Works Without OTEL
The new system gracefully degrades when OTEL is not available:

```python
# Even without opentelemetry installed, this works
from arshai.observability import get_llm_observability

observability = get_llm_observability()  # Returns no-op implementations
```

### Works With Any OTEL Setup
The new system automatically detects and uses existing OTEL configuration:

```python
# Your app sets up OTEL however you want
from opentelemetry import trace
from your_otel_setup import setup_otel

setup_otel()  # Your custom OTEL setup

# Arshai automatically uses your configuration
from arshai.observability import get_llm_observability

observability = get_llm_observability()  # Uses your OTEL setup
```

## Testing the Migration

### 1. Verify No Provider Creation
```python
import sys
from unittest.mock import patch

# This should not be called by Arshai
with patch('opentelemetry.trace.set_tracer_provider') as mock_set:
    from arshai.observability import get_llm_observability
    observability = get_llm_observability()
    
    # Should not have been called
    assert not mock_set.called, "Arshai should not create TracerProviders"
```

### 2. Test With and Without OTEL
```python
# Test without OTEL
import sys
sys.modules['opentelemetry'] = None

from arshai.observability import get_llm_observability
observability = get_llm_observability()  # Should work with no-ops

# Test with OTEL (after proper import)
from opentelemetry import trace
observability = get_llm_observability()  # Should use real OTEL
```

### 3. Verify Metrics
```python
async with observability.observe_llm_call(
    provider="test", 
    model="test-model", 
    method_name="test"
) as timing_data:
    timing_data.record_first_token()
    timing_data.record_token()
    await observability.record_usage_data(timing_data, {
        'input_tokens': 10,
        'output_tokens': 20,
        'total_tokens': 30
    })

# Verify timing data was recorded
assert timing_data.time_to_first_token is not None
assert timing_data.output_tokens == 20
```

## Key Benefits

1. **✅ OTEL Compliant**: Never creates providers, always uses `get_tracer()`
2. **✅ Non-Intrusive**: Respects parent application's OTEL setup
3. **✅ Graceful Degradation**: Works with and without OTEL
4. **✅ Package Isolation**: Uses package-specific configuration
5. **✅ LLM Optimized**: Designed specifically for LLM observability patterns
6. **✅ Production Ready**: Handles errors gracefully, no performance impact when disabled

## Common Issues

### Q: "My spans aren't showing up"
A: Check that your parent application has set up an OTEL TracerProvider with proper exporters.

### Q: "Metrics aren't being exported"
A: Ensure your parent application has configured an OTEL MeterProvider with appropriate metric exporters.

### Q: "Too much telemetry data"
A: Use `ARSHAI_TELEMETRY_LEVEL=ERROR` or disable specific features like `ARSHAI_LOG_PROMPTS=false`.

### Q: "Observability is disabled"
A: Check `ARSHAI_TELEMETRY_ENABLED=true` and verify OTEL is properly installed and configured.

## Migration Checklist

- [ ] Update imports to use new observability classes
- [ ] Replace `ObservabilityManager` with `get_llm_observability()`
- [ ] Update environment variables to use `ARSHAI_*` prefix
- [ ] Test that no OTEL providers are created by Arshai
- [ ] Verify observability works with your existing OTEL setup
- [ ] Test graceful degradation when OTEL is unavailable
- [ ] Update documentation and examples
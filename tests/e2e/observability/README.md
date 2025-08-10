# Arshai Observability End-to-End Test Suite

This directory contains a comprehensive end-to-end test suite for the Arshai observability system. It verifies that all 4 key metrics are properly collected, exported to real observability backends, and that there are no side effects on LLM calls.

## 🎯 What This Tests

### Key Metrics (As Requested)
- ✅ **`llm_time_to_first_token_seconds`** - Time from request start to first token
- ✅ **`llm_time_to_last_token_seconds`** - Time from request start to last token  
- ✅ **`llm_duration_first_to_last_token_seconds`** - Duration from first token to last token
- ✅ **`llm_completion_tokens`** - Count of completion tokens generated

### Full Stack Integration
- **OpenTelemetry Collector**: Receives and processes telemetry data
- **Jaeger**: Distributed tracing backend
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization dashboards

### LLM Provider Testing
- **OpenAI**: Complete observability integration
- **Azure OpenAI**: Token counting and metrics (if API key provided)
- **Anthropic Claude**: Native token counting (if API key provided)
- **Google Gemini**: Token estimation (if API key provided)

### Test Scenarios
- **Simple Chat Completion**: Basic request/response with timing
- **Streaming Completion**: Real-time token timing and chunk processing
- **Concurrent Requests**: Multiple parallel requests with individual tracking
- **Error Handling**: Graceful degradation when observability fails

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Arshai LLM    │───▶│ OTLP Collector  │───▶│    Backends     │
│  + Observability│    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              │                   ┌────▼────┐
                              │                   │ Jaeger  │
                              │                   │(Traces) │
                              │                   └─────────┘
                              │                        │
                              │                   ┌────▼────┐
                              └──────────────────▶│Prometheus│
                                                  │(Metrics)│
                                                  └─────────┘
                                                       │
                                                  ┌────▼────┐
                                                  │ Grafana │
                                                  │(Dashboards)│
                                                  └─────────┘
```

## 🚀 Quick Start

### Prerequisites

1. **Docker & Docker Compose**
   ```bash
   # Check if Docker is running
   docker info
   
   # Install if needed
   # macOS: brew install docker docker-compose
   # Ubuntu: apt install docker.io docker-compose
   ```

2. **Python 3.8+**
   ```bash
   python3 --version
   ```

3. **OpenAI API Key** (required)
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

4. **Optional API Keys** (for additional provider testing)
   ```bash
   export ANTHROPIC_API_KEY="your-anthropic-key"
   export GOOGLE_API_KEY="your-google-key"
   export AZURE_OPENAI_API_KEY="your-azure-key"
   ```

### Running the Test

```bash
# Navigate to test directory
cd tests/e2e/observability/

# Run the complete test suite
./run_test.sh
```

The script will:
1. ✅ Start all observability services (Docker Compose)
2. ✅ Wait for services to be ready
3. ✅ Install Python dependencies
4. ✅ Run comprehensive tests
5. ✅ Verify metrics and traces collection
6. ✅ Provide access URLs for manual inspection

## 📊 Expected Output

### Test Console Output
```
🚀 Arshai Observability End-to-End Test Suite
==============================================================

📋 PHASE 1: DEPENDENCY CHECKS
✅ Dependencies
✅ OTLP Collector
✅ Jaeger  
✅ Prometheus

📋 PHASE 2: PROVIDER TESTS
🧪 Testing openai provider with observability...
  📝 Testing simple chat completion...
  ✅ Simple completion: 45 tokens in 1.23s
  📝 Testing streaming completion...
  ✅ Streaming: 8 chunks, 52 tokens
  📝 Testing concurrent requests...
  ✅ Concurrent: 3/3 successful in 2.15s

📋 PHASE 3: OBSERVABILITY VERIFICATION
🔍 Verifying metrics collection...
  ✅ llm_time_to_first_token_seconds: 3 series
  ✅ llm_time_to_last_token_seconds: 3 series
  ✅ llm_duration_first_to_last_token_seconds: 3 series
  ✅ llm_completion_tokens: 3 series
✅ Metrics verification: 4/4 key metrics found

🔍 Verifying traces collection...
✅ Found 9 traces in Jaeger
  📊 Found 15 LLM-related spans

📋 PHASE 4: TEST SUMMARY
============================================================
📊 END-TO-END TEST SUMMARY
============================================================
Tests Run: 8
Tests Passed: 8
Tests Failed: 0
Success Rate: 100.0%
Providers Tested: openai

🎯 KEY METRICS STATUS:
  ✅ llm_time_to_first_token_seconds: 3 series
  ✅ llm_time_to_last_token_seconds: 3 series
  ✅ llm_duration_first_to_last_token_seconds: 3 series
  ✅ llm_completion_tokens: 3 series

💡 NEXT STEPS:
  1. View metrics: http://localhost:9090 (Prometheus)
  2. View traces: http://localhost:16686 (Jaeger)
  3. View dashboards: http://localhost:3000 (Grafana, admin/admin)
============================================================

🎉 END-TO-END TEST SUITE: PASSED
```

## 🔍 Manual Verification

After tests complete, you can manually verify the observability data:

### 1. Jaeger (Traces) - http://localhost:16686
- **Service**: `arshai-e2e-test`
- **Operations**: Look for `llm.chat_completion`, `llm.stream_completion`
- **Spans**: Each span should have timing attributes and token counts
- **Search**: Try searching by service or operation name

### 2. Prometheus (Metrics) - http://localhost:9090
Query these key metrics:
```promql
# Key metrics
llm_time_to_first_token_seconds
llm_time_to_last_token_seconds
llm_duration_first_to_last_token_seconds
llm_completion_tokens

# Additional metrics
llm_requests_total
llm_active_requests
llm_request_duration_seconds
```

### 3. Grafana (Dashboards) - http://localhost:3000
- **Login**: admin/admin
- **Dashboard**: "Arshai LLM Observability"
- **Panels**: Pre-configured panels for all key metrics
- **Time Range**: Last 30 minutes (adjust as needed)

## 🛠️ Manual Testing

You can also run individual components manually:

### Start Only Infrastructure
```bash
docker-compose up -d
```

### Run Only Python Tests
```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="your-key"

# Run tests
python test_e2e_observability.py
```

### Check Service Status
```bash
# Check all services
docker-compose ps

# Check logs
docker-compose logs otel-collector
docker-compose logs jaeger
docker-compose logs prometheus
docker-compose logs grafana
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Services Not Starting
```bash
# Check Docker is running
docker info

# Check ports are available
lsof -i :4317  # OTLP Collector
lsof -i :16686 # Jaeger
lsof -i :9090  # Prometheus
lsof -i :3000  # Grafana

# Restart services
docker-compose down
docker-compose up -d
```

#### 2. No Metrics in Prometheus
```bash
# Check OTLP Collector logs
docker-compose logs otel-collector

# Check if metrics endpoint is accessible
curl http://localhost:8889/metrics

# Verify Prometheus config
curl http://localhost:9090/api/v1/status/config
```

#### 3. No Traces in Jaeger
```bash
# Check Jaeger logs
docker-compose logs jaeger

# Check if Jaeger API is working
curl http://localhost:16686/api/services

# Verify OTLP export is working
curl http://localhost:4317  # Should connect
```

#### 4. Test Failures
```bash
# Run with debug logging
export PYTHONPATH=/path/to/arshai
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
import asyncio
from test_e2e_observability import main
asyncio.run(main())
"

# Check API keys are set
echo $OPENAI_API_KEY
```

### Debug Mode

Enable detailed logging by modifying the test:

```python
# In test_e2e_observability.py, change logging level
logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## 📁 File Structure

```
tests/e2e/observability/
├── README.md                     # This file
├── docker-compose.yml            # Full observability stack
├── otel-collector-config.yaml    # OTLP Collector configuration
├── prometheus.yml                # Prometheus configuration
├── grafana-datasources.yml       # Grafana data sources
├── grafana-dashboards.yml        # Grafana dashboard config
├── dashboards/
│   └── arshai-llm-metrics.json   # Pre-built LLM dashboard
├── test_config.yaml              # Test configuration
├── test_e2e_observability.py     # Main test script
├── requirements.txt              # Python dependencies
└── run_test.sh                   # Test runner script
```

## 🔧 Configuration

### Modifying Test Configuration

Edit `test_config.yaml` to customize:

```yaml
observability:
  # Change service name
  service_name: "my-custom-test"
  
  # Adjust export intervals for testing
  metric_export_interval: 5  # Faster for testing
  
  # Enable/disable providers
  provider_configs:
    openai:
      track_token_timing: true
    anthropic:
      track_token_timing: false  # Disable if no API key
```

### Custom Docker Compose

Modify `docker-compose.yml` to:
- Change port mappings
- Add custom environment variables
- Use different image versions
- Add additional services

## 🎯 Success Criteria

The test suite passes when:

1. **✅ All 4 key metrics are collected** and visible in Prometheus
2. **✅ Traces are exported** to Jaeger with proper span attributes
3. **✅ LLM calls work normally** without side effects
4. **✅ Multiple providers tested** (if API keys available)
5. **✅ Streaming scenarios work** with token-level timing
6. **✅ Concurrent requests** are individually tracked
7. **✅ Error scenarios** are handled gracefully

## 🚀 Next Steps

After successful testing:

1. **Production Setup**: Use the Docker Compose as a template for production
2. **Custom Dashboards**: Build custom Grafana dashboards for your use case
3. **Alerting**: Set up Prometheus alerts for key metrics
4. **Integration**: Integrate with your existing observability infrastructure
5. **Scaling**: Configure for high-throughput production workloads

## 🤝 Contributing

To extend the test suite:

1. **Add Providers**: Implement tests for additional LLM providers
2. **Add Metrics**: Test additional custom metrics
3. **Add Scenarios**: Test complex workflow scenarios
4. **Add Assertions**: Add more detailed validation logic
5. **Add Backends**: Test with other observability backends

The test suite is designed to be comprehensive yet extensible for your specific observability needs.
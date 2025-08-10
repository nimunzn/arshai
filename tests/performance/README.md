# Performance Testing Suite

Comprehensive production-grade load testing for Arshai framework performance optimizations.

## Quick Start

Run all tests with the performance suite:

```bash
# Quick validation (5 minutes)
python tests/performance/run_performance_suite.py --quick --report

# Full production test suite (2+ hours)
python tests/performance/run_performance_suite.py --full --report

# Individual test categories
python tests/performance/run_performance_suite.py --moderate --high --report
```

Run individual test files:

```bash
# Test HTTP connection pooling
pytest tests/performance/test_connection_pool_load.py -v

# Test thread pool management  
pytest tests/performance/test_thread_pool_load.py -v

# Test vector database async operations
pytest tests/performance/test_vector_db_load.py -v
```

## Test Categories

### 1. HTTP Connection Pooling (`test_connection_pool_load.py`)
- **Moderate Load**: 100 concurrent requests
- **High Load**: 500 concurrent requests  
- **Extreme Load**: 1,000 concurrent requests
- **Sustained Load**: 5 minutes continuous load
- **Recovery Tests**: Connection pool failure recovery

**Validates**: SearxNG connection pooling prevents container crashes

### 2. Thread Pool Management (`test_thread_pool_load.py`) 
- **Moderate Load**: 50 concurrent thread operations
- **High Load**: 200 concurrent thread operations
- **Extreme Load**: 500 concurrent thread operations
- **Sustained Load**: 2 minutes continuous operations
- **Deadlock Prevention**: Multi-threaded contention testing

**Validates**: MCP tool thread pooling prevents deadlocks

### 3. Vector Database Async (`test_vector_db_load.py`)
- **Moderate Load**: 100 concurrent vector searches
- **High Load**: 500 concurrent vector searches
- **Extreme Load**: 1,000 concurrent vector searches
- **Batch Operations**: High-volume document insertion
- **Mixed Workload**: 70% search, 30% insertion operations
- **Sustained Load**: 3 minutes continuous operations

**Validates**: Milvus async operations prevent event loop blocking

## Configuration

Set environment variables for testing:

```bash
# Connection pool limits
export ARSHAI_MAX_CONNECTIONS=100
export ARSHAI_MAX_CONNECTIONS_PER_HOST=20
export ARSHAI_CONNECTION_TIMEOUT=30

# Thread pool limits  
export ARSHAI_MAX_THREADS=32

# Memory limits
export ARSHAI_MAX_MEMORY_MB=4096
export ARSHAI_CLEANUP_INTERVAL=300
```

## Performance Thresholds

Tests validate these production thresholds:

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| Success Rate | ≥95% | Reliability under load |
| Connection Errors | 0 | No connection exhaustion |
| Average Response Time | ≤2000ms | Acceptable latency |
| P95 Response Time | ≤5000ms | Tail latency control |
| Memory Growth | ≤500MB | Memory leak prevention |
| Thread Creation Errors | 0 | No thread exhaustion |
| Event Loop Blocking | ≤500ms | Async operation validation |

## Quick Development Testing

Each test file supports quick testing:

```bash
# Quick connection pool test
python tests/performance/test_connection_pool_load.py --quick

# Quick thread pool test  
python tests/performance/test_thread_pool_load.py --quick

# Quick vector database test
python tests/performance/test_vector_db_load.py --quick
```

## Report Generation

The test suite generates:

1. **JSON Results**: Machine-readable metrics and timings
2. **Markdown Report**: Human-readable performance analysis
3. **Console Output**: Real-time test progress and results

Example report sections:
- Executive Summary with pass/fail counts
- Performance optimization validation status
- Detailed test results with metrics
- Production readiness assessment
- Deployment recommendations

## Integration with CI/CD

Add to your CI pipeline:

```yaml
- name: Run Performance Tests
  run: |
    python tests/performance/run_performance_suite.py --moderate --integration --metrics > performance_metrics.json
    
- name: Check Performance Thresholds
  run: |
    # Parse JSON and validate against thresholds
    python -c "
    import json
    with open('performance_metrics.json') as f:
        metrics = json.load(f)
    assert metrics['success_rate'] >= 95, f'Success rate {metrics[\"success_rate\"]}% < 95%'
    assert metrics['failed_tests'] == 0, f'{metrics[\"failed_tests\"]} tests failed'
    "
```

## Monitoring in Production

Use these metrics for production monitoring:

```promql
# Connection pool usage
arshai_http_connections_active / arshai_http_connections_limit * 100

# Thread pool usage  
arshai_thread_pool_active / arshai_thread_pool_limit * 100

# Vector operation latency
histogram_quantile(0.95, rate(arshai_vector_search_duration_seconds_bucket[5m]))
```

See `docs/09-deployment/performance-optimization.md` for complete monitoring setup.

## Troubleshooting

### Common Issues

**Connection Timeouts**:
```bash
# Increase connection limits
export ARSHAI_MAX_CONNECTIONS=200
export ARSHAI_MAX_CONNECTIONS_PER_HOST=30
```

**Thread Pool Exhaustion**:
```bash
# Increase thread limit
export ARSHAI_MAX_THREADS=64
```

**Memory Issues**:
```bash
# Increase memory limit
export ARSHAI_MAX_MEMORY_MB=8192
```

### Debug Mode

Enable debug logging:

```bash
export PYTHONPATH=/path/to/arshai
export ARSHAI_LOG_LEVEL=DEBUG
python tests/performance/run_performance_suite.py --quick --report
```

## Production Validation Checklist

Before deploying to production:

- [ ] All moderate load tests pass (≥95% success rate)
- [ ] High load tests pass (≥90% success rate)  
- [ ] No connection exhaustion errors
- [ ] No thread creation errors
- [ ] No event loop blocking detected
- [ ] Memory growth within acceptable limits
- [ ] Sustained load tests show stability
- [ ] Recovery tests demonstrate resilience

Run the full test suite for complete validation:

```bash
python tests/performance/run_performance_suite.py --full --report --metrics
```
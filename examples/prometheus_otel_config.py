#!/usr/bin/env python3
"""
Example configuration for collecting Arshai metrics in Prometheus via OTEL.

This example shows how to set up your parent application to properly export
OTEL metrics to Prometheus, ensuring Arshai's LLM metrics are collected.

For troubleshooting metrics collection issues in Prometheus/OTEL setup.
"""

import time
import asyncio
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

# Import Arshai observability
from arshai.observability import get_llm_observability, PackageObservabilityConfig, ObservabilityLevel


def setup_prometheus_direct():
    """Option 1: Direct Prometheus export (recommended for development)"""
    print("🔧 Setting up direct Prometheus metrics export...")
    
    # Create resource with service information
    resource = Resource.create({
        "service.name": "my-ai-app",
        "service.version": "1.0.0",
        "deployment.environment": "production"
    })
    
    # Set up Prometheus metrics reader
    prometheus_reader = PrometheusMetricReader()
    
    # Create meter provider with Prometheus reader
    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[prometheus_reader]
    )
    
    # Set global meter provider (Arshai will automatically detect this)
    metrics.set_meter_provider(meter_provider)
    
    print("✅ Prometheus metrics available at http://localhost:8000/metrics")
    print("   Add this endpoint to your Prometheus scrape config")


def setup_otlp_collector():
    """Option 2: OTLP collector (recommended for production)"""
    print("🔧 Setting up OTLP metrics export to collector...")
    
    # Create resource
    resource = Resource.create({
        "service.name": "my-ai-app", 
        "service.version": "1.0.0"
    })
    
    # Set up OTLP exporter to collector
    otlp_exporter = OTLPMetricExporter(
        endpoint="http://otel-collector:4317",  # Your collector endpoint
        insecure=True
    )
    
    # Create periodic reader for OTLP export
    otlp_reader = PeriodicExportingMetricReader(
        exporter=otlp_exporter,
        export_interval_millis=10000  # Export every 10 seconds
    )
    
    # Create meter provider
    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[otlp_reader]
    )
    
    # Set global meter provider
    metrics.set_meter_provider(meter_provider)
    
    print("✅ OTLP metrics exporting to collector")
    print("   Configure collector to export to Prometheus")


async def test_arshai_metrics():
    """Test that Arshai metrics are being collected properly"""
    print("\n📊 Testing Arshai LLM metrics collection...")
    
    # Create Arshai observability config
    config = PackageObservabilityConfig(
        enabled=True,
        level=ObservabilityLevel.INFO,
        collect_metrics=True,
        trace_llm_calls=True,
        track_token_timing=True
    )
    
    # Get observability instance (will use your OTEL setup)
    observability = get_llm_observability(config)
    
    if not observability.is_enabled():
        print("❌ Arshai observability not enabled")
        return
    
    if not observability.is_metrics_enabled():
        print("❌ Arshai metrics not enabled")
        return
    
    print("✅ Arshai observability is properly configured")
    print("✅ Arshai metrics collection is enabled")
    
    # Simulate LLM calls to generate metrics
    for i in range(3):
        print(f"   Simulating LLM call {i+1}...")
        
        async with observability.observe_llm_call(
            provider="test_provider",
            model="test_model",
            method_name="chat"
        ) as timing_data:
            # Simulate LLM processing time
            await asyncio.sleep(0.1)
            timing_data.record_first_token()
            
            await asyncio.sleep(0.05)
            timing_data.record_token()
            
            # Record usage
            await observability.record_usage_data(timing_data, {
                'input_tokens': 50,
                'output_tokens': 25,
                'total_tokens': 75
            })
    
    print("✅ Generated test LLM metrics")
    print("   Check your metrics endpoint for:")
    print("   - arshai_llm_time_to_first_token_seconds")
    print("   - arshai_llm_time_to_last_token_seconds")
    print("   - arshai_llm_request_duration_seconds")
    print("   - arshai_llm_token_usage_total")


def verify_metrics_collection():
    """Verify that metrics are being collected properly"""
    print("\n🔍 Verifying OTEL metrics setup...")
    
    try:
        # Check if meter provider is configured
        meter_provider = metrics.get_meter_provider()
        
        if hasattr(meter_provider, 'get_meter'):
            print("✅ MeterProvider is properly configured")
        else:
            print("❌ MeterProvider not found or using NoOp implementation")
            return False
        
        # Test creating a meter
        test_meter = meter_provider.get_meter("test", version="1.0.0")
        test_counter = test_meter.create_counter(
            name="test_metric",
            description="Test metric for verification"
        )
        
        # Record a test metric
        test_counter.add(1, {"test": "verification"})
        print("✅ Successfully created and recorded test metric")
        
        # Check if it's the Arshai meter
        arshai_meter = meter_provider.get_meter("arshai", version="1.2.3")
        print("✅ Arshai meter successfully created")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying metrics setup: {e}")
        return False


def print_troubleshooting_guide():
    """Print troubleshooting information for common issues"""
    print("\n" + "="*60)
    print("🔧 TROUBLESHOOTING GUIDE")
    print("="*60)
    
    print("\n📋 Common Issues & Solutions:")
    
    print("\n1. Metrics not appearing in Prometheus:")
    print("   ✅ Verify MeterProvider is set before importing Arshai")
    print("   ✅ Check Prometheus scrape config includes your metrics endpoint")
    print("   ✅ Ensure metrics reader is properly configured")
    print("   ✅ Wait for export interval (default: 10 seconds)")
    
    print("\n2. Arshai observability disabled:")
    print("   ✅ Check ARSHAI_TELEMETRY_ENABLED=true")
    print("   ✅ Check ARSHAI_COLLECT_METRICS=true")
    print("   ✅ Verify OTEL dependencies are installed")
    
    print("\n3. No metrics data recorded:")
    print("   ✅ Ensure LLM calls are being made through Arshai clients")
    print("   ✅ Check observability config is passed to LLM clients")
    print("   ✅ Verify no exceptions in metric recording")
    
    print("\n4. Collector setup issues:")
    print("   ✅ Verify collector endpoint is accessible")
    print("   ✅ Check collector configuration for Prometheus export")
    print("   ✅ Confirm network connectivity between services")
    
    print("\n📊 Expected Arshai Metrics:")
    print("   - arshai_llm_request_duration_seconds (histogram)")
    print("   - arshai_llm_time_to_first_token_seconds (histogram)")  
    print("   - arshai_llm_time_to_last_token_seconds (histogram)")
    print("   - arshai_llm_request_total (counter)")
    print("   - arshai_llm_request_errors_total (counter)")
    print("   - arshai_llm_active_requests (up_down_counter)")
    print("   - arshai_llm_token_usage_total (counter)")


async def main():
    """Main function to demonstrate OTEL metrics setup with Arshai"""
    print("🚀 Arshai Prometheus/OTEL Metrics Configuration Example")
    print("="*60)
    
    # Option 1: Set up direct Prometheus export (uncomment to use)
    setup_prometheus_direct()
    
    # Option 2: Set up OTLP collector export (uncomment to use)  
    # setup_otlp_collector()
    
    # Verify the setup
    if not verify_metrics_collection():
        print("\n❌ Metrics setup verification failed")
        print_troubleshooting_guide()
        return
    
    # Test Arshai metrics
    await test_arshai_metrics()
    
    # Wait a bit for metrics to be exported
    print("\n⏳ Waiting for metrics export...")
    time.sleep(15)
    
    print("\n✅ Setup complete!")
    print("Check your Prometheus/metrics endpoint for Arshai LLM metrics")
    
    # Print troubleshooting guide
    print_troubleshooting_guide()


if __name__ == "__main__":
    asyncio.run(main())
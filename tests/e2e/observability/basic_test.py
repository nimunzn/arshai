#!/usr/bin/env python3
"""
Basic test to validate observability components without external dependencies.
This creates configuration programmatically instead of loading from YAML.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

print(f"🔍 Testing Arshai observability setup...")
print(f"Project root: {project_root}")

def test_config_creation():
    """Test configuration creation programmatically."""
    try:
        from arshai.observability.config import ObservabilityConfig
        
        # Create config programmatically
        config = ObservabilityConfig(
            service_name="test-service",
            service_version="1.0.0",
            environment="test",
            track_token_timing=True,
            collect_metrics=True,
            trace_requests=True,
            otlp_endpoint="http://localhost:4317"
        )
        
        print("✅ ObservabilityConfig created successfully")
        print(f"  Service name: {config.service_name}")
        print(f"  Track token timing: {config.track_token_timing}")
        print(f"  OTLP endpoint: {config.otlp_endpoint}")
        
        # Test provider-specific settings
        openai_enabled = config.is_token_timing_enabled("openai")
        print(f"  OpenAI token timing: {openai_enabled}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_core_components():
    """Test core observability components."""
    try:
        from arshai.observability.core import ObservabilityManager
        from arshai.observability.config import ObservabilityConfig
        
        config = ObservabilityConfig(
            service_name="test-service",
            track_token_timing=True,
            collect_metrics=True,
            trace_requests=True
        )
        
        # Create manager (should work without OTLP dependencies)
        manager = ObservabilityManager(config)
        
        print("✅ ObservabilityManager created successfully")
        print(f"  Enabled: {manager.is_enabled()}")
        print(f"  Token timing enabled for openai: {manager.is_token_timing_enabled('openai')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Core components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics_components():
    """Test metrics collection components."""
    try:
        from arshai.observability.metrics import MetricsCollector, TimingData
        from arshai.observability.config import ObservabilityConfig
        
        config = ObservabilityConfig(
            service_name="test-service",
            collect_metrics=True
        )
        
        # Create metrics collector (should work without OTLP)
        collector = MetricsCollector(config)
        
        print("✅ MetricsCollector created successfully")
        print(f"  Enabled: {collector.is_enabled()}")
        
        # Test timing data
        timing = TimingData()
        timing.record_first_token()
        timing.record_token()
        timing.update_token_counts(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        
        print("✅ TimingData works correctly")
        print(f"  Time to first token: {timing.time_to_first_token}")
        print(f"  Total tokens: {timing.total_tokens}")
        
        # Test attributes creation
        attributes = collector.create_attributes("openai", "gpt-3.5-turbo", custom="value")
        print(f"  Attributes: {attributes}")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_factory_integration():
    """Test factory integration components."""
    try:
        from arshai.observability.factory_integration import ObservableFactory
        from arshai.observability.config import ObservabilityConfig
        
        config = ObservabilityConfig(
            service_name="test-service",
            track_token_timing=True
        )
        
        # Create a mock factory class
        class MockFactory:
            _providers = {"openai": "MockOpenAI"}
            
            @classmethod
            def create(cls, provider, config, **kwargs):
                return f"Mock{provider.title()}Client"
        
        # Create observable factory
        obs_factory = ObservableFactory(MockFactory, config)
        
        print("✅ ObservableFactory created successfully")
        print(f"  Observability enabled: {obs_factory.is_observability_enabled()}")
        print(f"  OpenAI enabled: {obs_factory.is_observability_enabled('openai')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Factory integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imports():
    """Test that all main observability components can be imported."""
    try:
        from arshai.observability import ObservabilityConfig, ObservabilityManager
        print("✅ Main imports work")
        
        from arshai.observability.metrics import MetricsCollector, TimingData
        print("✅ Metrics imports work")
        
        from arshai.observability.core import ObservabilityManager
        print("✅ Core imports work")
        
        from arshai.observability.factory_integration import ObservableFactory
        print("✅ Factory imports work")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("🧪 Arshai Observability Component Tests")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_imports),
        ("Configuration Creation", test_config_creation),
        ("Core Components", test_core_components),
        ("Metrics Components", test_metrics_components),
        ("Factory Integration", test_factory_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All component tests passed!")
        print("\n📁 Test suite structure is ready:")
        print("  📄 docker-compose.yml - Full observability stack")
        print("  📄 test_config.yaml - Configuration file")
        print("  📄 test_e2e_observability.py - Comprehensive end-to-end tests")
        print("  📄 run_test.sh - Test runner script")
        print("\n💡 To run full end-to-end tests:")
        print("  1. Set environment variable: export OPENAI_API_KEY='your-key'")
        print("  2. Install dependencies: pip install -r requirements.txt")
        print("  3. Start services: docker-compose up -d")
        print("  4. Run tests: ./run_test.sh")
        return True
    else:
        print("❌ Some tests failed!")
        print("Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
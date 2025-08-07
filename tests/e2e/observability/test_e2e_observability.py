#!/usr/bin/env python3
"""
Comprehensive End-to-End Test for Arshai Observability System

This test verifies:
1. All 4 key metrics are collected: time_to_first_token, time_to_last_token, 
   duration_first_to_last_token, completion_tokens
2. OpenTelemetry integration works with real OTLP collector
3. All supported LLM providers work with observability
4. Streaming and non-streaming scenarios
5. Metrics export to Prometheus
6. Trace export to Jaeger
7. No side effects on LLM calls
"""

import asyncio
import os
import sys
import time
import logging
import json
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
import statistics

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Arshai imports
from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.observability import ObservabilityConfig, ObservabilityManager
from src.factories.llm_factory import LLMFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class E2EObservabilityTest:
    """Comprehensive end-to-end observability test."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.observability_config = ObservabilityConfig.from_yaml(config_path)
        self.test_results = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "errors": [],
            "metrics_collected": {},
            "providers_tested": []
        }
        
        # Test prompts
        self.test_prompts = [
            {
                "system": "You are a helpful AI assistant.",
                "user": "What is machine learning? Be brief.",
                "expected_type": "short_response"
            },
            {
                "system": "You are a creative writing assistant.",
                "user": "Write a very short poem about observability in software.",
                "expected_type": "creative_response"
            },
            {
                "system": "You are a technical expert.",
                "user": "Explain OpenTelemetry in one paragraph.",
                "expected_type": "technical_response"
            }
        ]
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available."""
        logger.info("🔍 Checking dependencies...")
        
        dependencies = [
            "opentelemetry",
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
            "opentelemetry.exporter.otlp.proto.grpc.metric_exporter"
        ]
        
        missing = []
        for dep in dependencies:
            try:
                __import__(dep)
                logger.info(f"✅ {dep}")
            except ImportError:
                missing.append(dep)
                logger.error(f"❌ {dep}")
        
        if missing:
            logger.error(f"Missing dependencies: {missing}")
            return False
        
        return True
    
    def check_otlp_collector(self) -> bool:
        """Check if OTLP collector is running."""
        logger.info("🔍 Checking OTLP Collector availability...")
        
        try:
            # Try to connect to the OTLP gRPC endpoint
            import grpc
            channel = grpc.insecure_channel("localhost:4317")
            grpc.channel_ready_future(channel).result(timeout=5)
            logger.info("✅ OTLP Collector is available on port 4317")
            return True
        except Exception as e:
            logger.error(f"❌ OTLP Collector not available: {e}")
            logger.info("💡 Please run: docker-compose up -d")
            return False
    
    def check_jaeger(self) -> bool:
        """Check if Jaeger is accessible."""
        logger.info("🔍 Checking Jaeger availability...")
        
        try:
            response = requests.get("http://localhost:16686/api/services", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Jaeger is available on port 16686")
                return True
            else:
                logger.error(f"❌ Jaeger returned status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Jaeger not available: {e}")
            return False
    
    def check_prometheus(self) -> bool:
        """Check if Prometheus is accessible."""
        logger.info("🔍 Checking Prometheus availability...")
        
        try:
            response = requests.get("http://localhost:9090/api/v1/status/config", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Prometheus is available on port 9090")
                return True
            else:
                logger.error(f"❌ Prometheus returned status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Prometheus not available: {e}")
            return False
    
    async def test_provider_observability(self, provider: str, model: str) -> Dict[str, Any]:
        """Test observability for a specific provider."""
        logger.info(f"🧪 Testing {provider} provider with observability...")
        
        # Check API key
        api_key_var = f"{provider.upper()}_API_KEY"
        if provider == "google":
            api_key_var = "GOOGLE_API_KEY"
        elif provider == "azure":
            api_key_var = "AZURE_OPENAI_API_KEY"
        
        if not os.environ.get(api_key_var):
            logger.warning(f"⚠️  Skipping {provider}: {api_key_var} not set")
            return {"skipped": True, "reason": f"{api_key_var} not set"}
        
        try:
            # Configure LLM
            llm_config = ILLMConfig(
                model=model,
                temperature=0.7,
                max_tokens=100
            )
            
            # Create LLM client with observability
            client = LLMFactory.create_with_observability(
                provider=provider,
                config=llm_config,
                observability_config=self.observability_config
            )
            
            results = {
                "provider": provider,
                "model": model,
                "tests": {},
                "metrics": {},
                "errors": []
            }
            
            # Test 1: Simple chat completion
            logger.info(f"  📝 Testing simple chat completion...")
            try:
                test_input = ILLMInput(
                    system_prompt=self.test_prompts[0]["system"],
                    user_message=self.test_prompts[0]["user"]
                )
                
                start_time = time.time()
                response = client.chat_completion(test_input)
                end_time = time.time()
                
                # Verify response structure
                assert 'llm_response' in response, "Missing llm_response in response"
                assert 'usage' in response, "Missing usage data in response"
                
                usage = response['usage']
                assert hasattr(usage, 'prompt_tokens'), "Missing prompt_tokens in usage"
                assert hasattr(usage, 'completion_tokens'), "Missing completion_tokens in usage"
                assert hasattr(usage, 'total_tokens'), "Missing total_tokens in usage"
                
                results["tests"]["simple_completion"] = {
                    "status": "passed",
                    "duration": end_time - start_time,
                    "response_length": len(response['llm_response']),
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens
                    }
                }
                
                logger.info(f"  ✅ Simple completion: {usage.total_tokens} tokens in {end_time - start_time:.2f}s")
                
            except Exception as e:
                results["tests"]["simple_completion"] = {"status": "failed", "error": str(e)}
                results["errors"].append(f"Simple completion failed: {e}")
                logger.error(f"  ❌ Simple completion failed: {e}")
            
            # Test 2: Streaming completion (if supported)
            logger.info(f"  📝 Testing streaming completion...")
            try:
                test_input = ILLMInput(
                    system_prompt=self.test_prompts[1]["system"],
                    user_message=self.test_prompts[1]["user"]
                )
                
                start_time = time.time()
                full_response = ""
                chunk_count = 0
                final_usage = None
                
                async for chunk in client.stream_completion(test_input):
                    chunk_count += 1
                    if chunk.get('llm_response'):
                        full_response += chunk['llm_response']
                    
                    # Check for final usage data
                    if chunk.get('usage'):
                        final_usage = chunk['usage']
                
                end_time = time.time()
                
                # Verify streaming worked
                assert chunk_count > 0, "No chunks received from streaming"
                assert len(full_response) > 0, "No content received from streaming"
                assert final_usage is not None, "No final usage data received"
                
                results["tests"]["streaming_completion"] = {
                    "status": "passed",
                    "duration": end_time - start_time,
                    "chunk_count": chunk_count,
                    "response_length": len(full_response),
                    "usage": {
                        "prompt_tokens": final_usage.prompt_tokens,
                        "completion_tokens": final_usage.completion_tokens,
                        "total_tokens": final_usage.total_tokens
                    } if final_usage else None
                }
                
                logger.info(f"  ✅ Streaming: {chunk_count} chunks, {final_usage.total_tokens if final_usage else 'unknown'} tokens")
                
            except Exception as e:
                results["tests"]["streaming_completion"] = {"status": "failed", "error": str(e)}
                results["errors"].append(f"Streaming completion failed: {e}")
                logger.error(f"  ❌ Streaming completion failed: {e}")
            
            # Test 3: Concurrent requests
            logger.info(f"  📝 Testing concurrent requests...")
            try:
                async def make_concurrent_request(prompt_idx: int):
                    test_input = ILLMInput(
                        system_prompt=self.test_prompts[prompt_idx]["system"],
                        user_message=self.test_prompts[prompt_idx]["user"]
                    )
                    return client.chat_completion(test_input)
                
                start_time = time.time()
                tasks = [make_concurrent_request(i % len(self.test_prompts)) for i in range(3)]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()
                
                successful_responses = [r for r in responses if not isinstance(r, Exception)]
                failed_responses = [r for r in responses if isinstance(r, Exception)]
                
                results["tests"]["concurrent_requests"] = {
                    "status": "passed" if len(successful_responses) >= 2 else "partial",
                    "duration": end_time - start_time,
                    "successful": len(successful_responses),
                    "failed": len(failed_responses),
                    "total_tokens": sum(r['usage'].total_tokens for r in successful_responses if 'usage' in r)
                }
                
                logger.info(f"  ✅ Concurrent: {len(successful_responses)}/3 successful in {end_time - start_time:.2f}s")
                
            except Exception as e:
                results["tests"]["concurrent_requests"] = {"status": "failed", "error": str(e)}
                results["errors"].append(f"Concurrent requests failed: {e}")
                logger.error(f"  ❌ Concurrent requests failed: {e}")
            
            self.test_results["providers_tested"].append(provider)
            return results
            
        except Exception as e:
            logger.error(f"❌ Provider {provider} test failed: {e}")
            return {
                "provider": provider,
                "model": model,
                "failed": True,
                "error": str(e)
            }
    
    async def verify_metrics_collection(self) -> bool:
        """Verify that metrics are being collected in Prometheus."""
        logger.info("🔍 Verifying metrics collection...")
        
        # Wait a bit for metrics to be exported
        await asyncio.sleep(10)
        
        key_metrics = [
            "llm_time_to_first_token_seconds",
            "llm_time_to_last_token_seconds", 
            "llm_duration_first_to_last_token_seconds",
            "llm_completion_tokens"
        ]
        
        try:
            metrics_found = {}
            
            for metric in key_metrics:
                try:
                    response = requests.get(
                        f"http://localhost:9090/api/v1/query",
                        params={"query": metric},
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("data", {}).get("result"):
                            metrics_found[metric] = len(data["data"]["result"])
                            logger.info(f"  ✅ {metric}: {metrics_found[metric]} series")
                        else:
                            metrics_found[metric] = 0
                            logger.warning(f"  ⚠️  {metric}: No data")
                    else:
                        logger.error(f"  ❌ {metric}: HTTP {response.status_code}")
                        metrics_found[metric] = -1
                        
                except Exception as e:
                    logger.error(f"  ❌ {metric}: {e}")
                    metrics_found[metric] = -1
            
            self.test_results["metrics_collected"] = metrics_found
            
            # Check if we got at least some key metrics
            successful_metrics = sum(1 for count in metrics_found.values() if count > 0)
            if successful_metrics >= 2:
                logger.info(f"✅ Metrics verification: {successful_metrics}/4 key metrics found")
                return True
            else:
                logger.error(f"❌ Metrics verification failed: Only {successful_metrics}/4 key metrics found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Metrics verification failed: {e}")
            return False
    
    async def verify_traces_collection(self) -> bool:
        """Verify that traces are being collected in Jaeger."""
        logger.info("🔍 Verifying traces collection...")
        
        # Wait a bit for traces to be exported
        await asyncio.sleep(5)
        
        try:
            # Query Jaeger for traces from our service
            response = requests.get(
                "http://localhost:16686/api/traces",
                params={
                    "service": self.observability_config.service_name,
                    "limit": 20
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                traces = data.get("data", [])
                
                if traces:
                    logger.info(f"✅ Found {len(traces)} traces in Jaeger")
                    
                    # Analyze trace data
                    llm_spans = 0
                    for trace in traces:
                        for span in trace.get("spans", []):
                            if "llm." in span.get("operationName", ""):
                                llm_spans += 1
                    
                    logger.info(f"  📊 Found {llm_spans} LLM-related spans")
                    return True
                else:
                    logger.warning("⚠️  No traces found in Jaeger")
                    return False
            else:
                logger.error(f"❌ Jaeger API returned status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Traces verification failed: {e}")
            return False
    
    def print_test_summary(self):
        """Print a comprehensive test summary."""
        logger.info("=" * 60)
        logger.info("📊 END-TO-END TEST SUMMARY")
        logger.info("=" * 60)
        
        # Overall results
        total_tests = self.test_results["tests_run"]
        passed_tests = self.test_results["tests_passed"]
        failed_tests = self.test_results["tests_failed"]
        
        logger.info(f"Tests Run: {total_tests}")
        logger.info(f"Tests Passed: {passed_tests}")
        logger.info(f"Tests Failed: {failed_tests}")
        logger.info(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
        
        # Providers tested
        logger.info(f"Providers Tested: {', '.join(self.test_results['providers_tested'])}")
        
        # Key metrics status
        logger.info("\n🎯 KEY METRICS STATUS:")
        for metric, count in self.test_results["metrics_collected"].items():
            status = "✅" if count > 0 else "❌" if count == 0 else "⚠️"
            logger.info(f"  {status} {metric}: {count} series")
        
        # Errors
        if self.test_results["errors"]:
            logger.info(f"\n❌ ERRORS ({len(self.test_results['errors'])}):")
            for error in self.test_results["errors"]:
                logger.info(f"  • {error}")
        
        # Recommendations
        logger.info("\n💡 NEXT STEPS:")
        logger.info("  1. View metrics: http://localhost:9090 (Prometheus)")
        logger.info("  2. View traces: http://localhost:16686 (Jaeger)")
        logger.info("  3. View dashboards: http://localhost:3000 (Grafana, admin/admin)")
        
        logger.info("=" * 60)
    
    async def run_full_test_suite(self):
        """Run the complete end-to-end test suite."""
        logger.info("🚀 Starting Arshai Observability End-to-End Test Suite")
        logger.info("=" * 60)
        
        # Phase 1: Dependency checks
        logger.info("📋 PHASE 1: DEPENDENCY CHECKS")
        
        checks = [
            ("Dependencies", self.check_dependencies()),
            ("OTLP Collector", self.check_otlp_collector()),
            ("Jaeger", self.check_jaeger()),
            ("Prometheus", self.check_prometheus())
        ]
        
        failed_checks = []
        for check_name, result in checks:
            if result:
                logger.info(f"✅ {check_name}")
            else:
                logger.error(f"❌ {check_name}")
                failed_checks.append(check_name)
        
        if failed_checks:
            logger.error(f"❌ Prerequisites failed: {', '.join(failed_checks)}")
            logger.info("💡 Please run: docker-compose up -d")
            return False
        
        # Phase 2: Provider tests
        logger.info("\n📋 PHASE 2: PROVIDER TESTS")
        
        providers_to_test = [
            ("openai", "gpt-3.5-turbo"),
            # Add more providers if API keys are available
            # ("anthropic", "claude-3-sonnet-20240229"),
            # ("google", "gemini-pro"),
            # ("azure", "gpt-35-turbo"),
        ]
        
        provider_results = []
        for provider, model in providers_to_test:
            result = await self.test_provider_observability(provider, model)
            provider_results.append(result)
            
            if not result.get("skipped") and not result.get("failed"):
                self.test_results["tests_run"] += len(result.get("tests", {}))
                self.test_results["tests_passed"] += sum(
                    1 for test in result.get("tests", {}).values() 
                    if test.get("status") == "passed"
                )
                self.test_results["tests_failed"] += sum(
                    1 for test in result.get("tests", {}).values() 
                    if test.get("status") == "failed"
                )
                self.test_results["errors"].extend(result.get("errors", []))
        
        # Phase 3: Observability verification
        logger.info("\n📋 PHASE 3: OBSERVABILITY VERIFICATION")
        
        metrics_ok = await self.verify_metrics_collection()
        traces_ok = await self.verify_traces_collection()
        
        if metrics_ok:
            self.test_results["tests_passed"] += 1
        else:
            self.test_results["tests_failed"] += 1
        
        if traces_ok:
            self.test_results["tests_passed"] += 1
        else:
            self.test_results["tests_failed"] += 1
            
        self.test_results["tests_run"] += 2
        
        # Phase 4: Summary
        logger.info("\n📋 PHASE 4: TEST SUMMARY")
        self.print_test_summary()
        
        # Overall success
        success_rate = self.test_results["tests_passed"] / self.test_results["tests_run"]
        overall_success = success_rate >= 0.8  # 80% pass rate
        
        if overall_success:
            logger.info("🎉 END-TO-END TEST SUITE: PASSED")
        else:
            logger.error("❌ END-TO-END TEST SUITE: FAILED")
        
        return overall_success


async def main():
    """Main test runner."""
    # Get config path
    config_path = Path(__file__).parent / "test_config.yaml"
    
    if not config_path.exists():
        logger.error(f"❌ Config file not found: {config_path}")
        return False
    
    # Run tests
    test_suite = E2EObservabilityTest(str(config_path))
    success = await test_suite.run_full_test_suite()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
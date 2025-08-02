#!/usr/bin/env python3
"""Quick service verification script."""

import requests
import sys

def check_service(name, url, expected=None):
    """Check if a service is responding."""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ {name}: Running (HTTP {response.status_code})")
            if expected and expected in response.text:
                print(f"   → Content contains: {expected}")
            return True
        else:
            print(f"⚠️  {name}: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {name}: {e}")
        return False

def main():
    print("🔍 Checking observability services...")
    print("=" * 50)
    
    services = [
        ("Jaeger API", "http://localhost:16686/api/services"),
        ("Prometheus Config", "http://localhost:9090/api/v1/status/config"),
        ("Grafana Health", "http://localhost:3000/api/health"),
        ("OTLP Collector Metrics", "http://localhost:8889/metrics"),
    ]
    
    results = []
    for name, url in services:
        results.append(check_service(name, url))
    
    print("=" * 50)
    success_count = sum(results)
    total_count = len(results)
    
    print(f"📊 Services Status: {success_count}/{total_count} healthy")
    
    if success_count >= 3:  # At least 3 out of 4 services working
        print("🎉 Observability stack is ready for testing!")
        print("\n🔗 Access URLs:")
        print("  • Jaeger (Traces): http://localhost:16686")
        print("  • Prometheus (Metrics): http://localhost:9090")
        print("  • Grafana (Dashboards): http://localhost:3000 (admin/admin)")
        print("  • OTLP Collector gRPC: localhost:4317")
        print("  • OTLP Collector HTTP: localhost:4318")
        return True
    else:
        print("❌ Some services are not responding correctly.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
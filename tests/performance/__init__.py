"""
Performance Testing Suite for Arshai Framework

This module contains production-grade load tests to validate the performance
optimizations implemented in the Arshai framework under extreme load conditions.

Tests cover:
- HTTP connection pooling under 1000+ concurrent requests
- Thread pool limits under 500+ concurrent operations
- Vector database async operations under high load
- Memory management under sustained load
- Real-world production scenarios

All tests are designed to run against actual infrastructure to validate
production readiness and identify performance bottlenecks.
"""
#!/bin/bash

set -e

echo "🚀 Arshai Observability End-to-End Test Runner"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: docker-compose.yml not found. Please run from the test directory."
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Stop any existing containers
echo "🧹 Cleaning up existing containers..."
docker-compose down --remove-orphans || true

# Start the observability stack
echo "🐳 Starting observability stack..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are healthy
echo "🔍 Checking service health..."

# Check OTLP Collector
if curl -f http://localhost:8888/metrics > /dev/null 2>&1; then
    echo "✅ OTLP Collector is ready"
else
    echo "❌ OTLP Collector is not ready"
    docker-compose logs otel-collector
    exit 1
fi

# Check Jaeger
if curl -f http://localhost:16686/api/services > /dev/null 2>&1; then
    echo "✅ Jaeger is ready"
else
    echo "❌ Jaeger is not ready"
    docker-compose logs jaeger
    exit 1
fi

# Check Prometheus
if curl -f http://localhost:9090/api/v1/status/config > /dev/null 2>&1; then
    echo "✅ Prometheus is ready"
else
    echo "❌ Prometheus is not ready"
    docker-compose logs prometheus
    exit 1
fi

# Check Grafana
if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
    echo "✅ Grafana is ready"
else
    echo "❌ Grafana is not ready"
    docker-compose logs grafana
    exit 1
fi

echo ""
echo "🎯 All services are ready!"
echo ""
echo "📊 Service URLs:"
echo "  • Jaeger (Traces): http://localhost:16686"
echo "  • Prometheus (Metrics): http://localhost:9090"
echo "  • Grafana (Dashboards): http://localhost:3000 (admin/admin)"
echo "  • OTLP Collector: http://localhost:4317 (gRPC), http://localhost:4318 (HTTP)"
echo ""

# Install Python dependencies if needed
if [ ! -d "venv" ]; then
    echo "🐍 Setting up Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "🐍 Using existing virtual environment..."
    source venv/bin/activate
fi

# Check for required environment variables
echo "🔑 Checking environment variables..."
REQUIRED_VARS=("OPENAI_API_KEY")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    else
        echo "✅ $var is set"
    fi
done

# Optional environment variables
OPTIONAL_VARS=("ANTHROPIC_API_KEY" "GOOGLE_API_KEY" "AZURE_OPENAI_API_KEY")
for var in "${OPTIONAL_VARS[@]}"; do
    if [ -n "${!var}" ]; then
        echo "✅ $var is set (optional)"
    else
        echo "⚠️  $var not set (provider will be skipped)"
    fi
done

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    echo ""
    echo "❌ Missing required environment variables:"
    for var in "${MISSING_VARS[@]}"; do
        echo "  • $var"
    done
    echo ""
    echo "Please set them and run again:"
    echo "  export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

echo ""
echo "🧪 Running end-to-end observability tests..."
echo "=============================================="

# Run the test
python test_e2e_observability.py

TEST_EXIT_CODE=$?

echo ""
echo "=============================================="

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED!"
    echo ""
    echo "🔍 Next steps:"
    echo "  1. View traces: http://localhost:16686"
    echo "  2. View metrics: http://localhost:9090"
    echo "  3. View dashboards: http://localhost:3000"
    echo ""
    echo "🛑 To stop services: docker-compose down"
else
    echo "❌ SOME TESTS FAILED!"
    echo ""
    echo "🔍 Debugging:"
    echo "  1. Check logs: docker-compose logs"
    echo "  2. View services: docker-compose ps"
    echo "  3. Check traces: http://localhost:16686"
    echo "  4. Check metrics: http://localhost:9090"
fi

exit $TEST_EXIT_CODE
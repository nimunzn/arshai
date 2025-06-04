#!/bin/bash

# Build documentation script for Arshai package
# Usage: ./scripts/build_docs.sh [--watch]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCS_DIR="$PROJECT_ROOT/docs_sphinx"

echo "Building Arshai documentation..."

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Warning: No virtual environment detected. Consider using 'poetry shell' or activating a venv."
fi

# Install package in development mode
echo "Installing package in development mode..."
cd "$PROJECT_ROOT"
pip install -e .

# Install documentation dependencies
echo "Installing documentation dependencies..."
pip install -r "$DOCS_DIR/requirements.txt"

# Generate API documentation
echo "Generating API documentation..."
python "$PROJECT_ROOT/scripts/generate_api_docs.py"

# Build documentation
echo "Building HTML documentation..."
cd "$DOCS_DIR"

if [[ "$1" == "--watch" ]]; then
    echo "Starting documentation server with auto-reload..."
    echo "Documentation will be available at http://localhost:8000"
    echo "Press Ctrl+C to stop."
    
    # Install sphinx-autobuild if not present
    pip install sphinx-autobuild
    
    sphinx-autobuild . _build/html --host 0.0.0.0 --port 8000 --open-browser
else
    make clean
    make html
    
    echo ""
    echo "Documentation built successfully!"
    echo "Open: file://$DOCS_DIR/_build/html/index.html"
fi
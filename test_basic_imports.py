#!/usr/bin/env python3
"""
Test basic imports without optional dependencies.
"""

import subprocess
import sys
import tempfile
import os
from pathlib import Path

def test_basic_imports():
    """Test basic imports that don't require optional dependencies."""
    
    print("Testing basic package imports...")
    
    # Get the built wheel file
    dist_dir = Path("dist")
    wheel_files = list(dist_dir.glob("*.whl"))
    
    if not wheel_files:
        print("❌ No wheel file found in dist/")
        return False
    
    wheel_file = wheel_files[0]
    print(f"Found wheel: {wheel_file}")
    
    # Create a temporary virtual environment
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        venv_path = temp_path / "test_venv"
        
        # Create virtual environment
        print("Creating test virtual environment...")
        result = subprocess.run([
            sys.executable, "-m", "venv", str(venv_path)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to create venv: {result.stderr}")
            return False
        
        # Determine python executable in venv
        if os.name == 'nt':  # Windows
            python_exe = venv_path / "Scripts" / "python.exe"
        else:  # Unix/Linux/macOS
            python_exe = venv_path / "bin" / "python"
        
        # Install base dependencies first
        print("Installing base dependencies...")
        result = subprocess.run([
            str(python_exe), "-m", "pip", "install", "pydantic", "pyyaml", "aiohttp", "requests"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False
        
        # Install the package
        print(f"Installing package: {wheel_file}")
        result = subprocess.run([
            str(python_exe), "-m", "pip", "install", str(wheel_file)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to install package: {result.stderr}")
            return False
        
        # Test importing the package (basic imports only)
        test_code = '''
try:
    import arshai
    print(f"✓ arshai version: {arshai.__version__}")
except Exception as e:
    print(f"❌ Failed to import arshai: {e}")
    exit(1)

try:
    # Test core interfaces
    from arshai.core.interfaces import IAgent, ITool
    print("✓ Core interfaces imported")
except Exception as e:
    print(f"❌ Failed to import interfaces: {e}")
    exit(1)

try:
    # Test extension system
    from arshai.extensions import Plugin, PluginRegistry
    print("✓ Extension system imported")
except Exception as e:
    print(f"❌ Failed to import extensions: {e}")
    exit(1)

print("✅ Basic imports successful!")
'''
        
        print("Testing package imports...")
        result = subprocess.run([
            str(python_exe), "-c", test_code
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Import test failed: {result.stderr}")
            print(f"Stdout: {result.stdout}")
            return False
        
        print(result.stdout)
        print("✅ Package basic imports test passed!")
        return True

if __name__ == "__main__":
    success = test_basic_imports()
    sys.exit(0 if success else 1)
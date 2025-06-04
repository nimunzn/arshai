#!/usr/bin/env python3
"""
Test script to verify the built package works correctly.
"""

import subprocess
import sys
import tempfile
import os
from pathlib import Path

def test_package_installation():
    """Test that the package can be installed and imported."""
    
    print("Testing package installation...")
    
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
        
        # Install the package
        print(f"Installing package: {wheel_file}")
        result = subprocess.run([
            str(python_exe), "-m", "pip", "install", str(wheel_file)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to install package: {result.stderr}")
            return False
        
        # Test importing the package
        test_code = '''
import arshai
print(f"✓ arshai version: {arshai.__version__}")

# Test core imports
from arshai import Settings
print("✓ Settings imported")

from arshai.core.interfaces import IAgent, ITool
print("✓ Core interfaces imported")

# Test extension system
from arshai.extensions import Plugin, PluginRegistry
print("✓ Extension system imported")

print("✅ All imports successful!")
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
        print("✅ Package installation test passed!")
        return True

if __name__ == "__main__":
    success = test_package_installation()
    sys.exit(0 if success else 1)
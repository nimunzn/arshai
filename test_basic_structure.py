#!/usr/bin/env python3
"""Basic test to verify package structure without optional dependencies."""

import sys
import os

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test basic package imports."""
    print("Testing basic package structure...")
    
    try:
        # Test package import
        import arshai
        print(f"✓ arshai version: {arshai.__version__}")
        
        # Test core interfaces  
        from arshai.core.interfaces import IAgent, IAgentConfig, ITool, IWorkflow
        print("✓ Core interfaces imported")
        
        # Test that files were copied correctly
        from arshai.agents import conversation
        print("✓ Agents module accessible")
        
        from arshai.workflows import node
        print("✓ Workflows module accessible")
        
        # Test memory types
        from arshai.memory.memory_types import MemoryQuery
        print("✓ Memory types accessible")
        
        print("\n✅ Basic structure test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_imports()
    sys.exit(0 if success else 1)
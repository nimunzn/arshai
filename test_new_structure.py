#!/usr/bin/env python3
"""Test script to verify the new package structure works correctly."""

import sys
import traceback

def test_imports():
    """Test that basic imports work with the new structure."""
    print("Testing new package imports...")
    
    try:
        # Test main package import
        import arshai
        print("✓ import arshai")
        
        # Test version
        print(f"✓ arshai.__version__ = {arshai.__version__}")
        
        # Test Settings import
        from arshai import Settings
        print("✓ from arshai import Settings")
        
        # Test core interfaces
        from arshai.core.interfaces import (
            IAgent, IAgentConfig, IWorkflow, ITool
        )
        print("✓ from arshai.core.interfaces import ...")
        
        # Test compatibility layer
        from arshai.compat import enable_compatibility_mode
        print("✓ from arshai.compat import enable_compatibility_mode")
        
        print("\n✅ All basic imports successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_compatibility():
    """Test backward compatibility with old imports."""
    print("\n\nTesting backward compatibility...")
    
    try:
        # Enable compatibility mode
        from arshai.compat import enable_compatibility_mode
        enable_compatibility_mode()
        
        # Test old style imports (should work with warnings)
        print("\nTesting old-style imports (should show deprecation warnings):")
        
        # This would work after we copy the actual files
        # from arshai.core.interfaces import IAgent
        # print("✓ from arshai.core.interfaces import IAgent")
        
        print("✓ Compatibility mode enabled")
        return True
        
    except Exception as e:
        print(f"\n❌ Compatibility test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Arshai Package Structure")
    print("=" * 60)
    
    # Add package to path for testing
    sys.path.insert(0, '.')
    
    success = test_imports()
    # test_compatibility()  # Will test after file migration
    
    if success:
        print("\n✅ Package structure test passed!")
    else:
        print("\n❌ Package structure test failed!")
        sys.exit(1)
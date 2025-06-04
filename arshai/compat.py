"""
Compatibility layer for smooth migration from old import paths to new structure.

This module provides import aliases to maintain backward compatibility while
transitioning to the new package structure.
"""

import warnings
import sys
from typing import Any

class CompatibilityImporter:
    """Custom importer to handle old import paths and redirect to new ones."""
    
    # Mapping of old paths to new paths
    PATH_MAPPING = {
        'seedwork.interfaces': 'arshai.core.interfaces',
        'src.agents': 'arshai.agents',
        'src.workflows': 'arshai.workflows',
        'src.memory': 'arshai.memory',
        'src.tools': 'arshai.tools',
        'src.llms': 'arshai.llms',
        'src.embeddings': 'arshai.embeddings',
        'src.document_loaders': 'arshai.document_loaders',
        'src.vector_db': 'arshai.vector_db',
        'src.config': 'arshai.config',
        'src.utils': 'arshai.utils',
        'src.factories': 'arshai.factories',
        'src.callbacks': 'arshai.callbacks',
        'src.prompts': 'arshai.prompts',
        'src.rerankers': 'arshai.rerankers',
        'src.speech': 'arshai.speech',
        'src.web_search': 'arshai.web_search',
        'src.clients': 'arshai.clients',
    }
    
    @classmethod
    def install(cls):
        """Install the compatibility importer."""
        # Add to sys.meta_path if not already present
        if not any(isinstance(importer, cls) for importer in sys.meta_path):
            sys.meta_path.insert(0, cls())
    
    def find_module(self, fullname: str, path: Any = None):
        """Find module using old import paths."""
        for old_path, new_path in self.PATH_MAPPING.items():
            if fullname.startswith(old_path):
                return self
        return None
    
    def load_module(self, fullname: str):
        """Load module with deprecation warning."""
        for old_path, new_path in self.PATH_MAPPING.items():
            if fullname.startswith(old_path):
                # Calculate new module name
                new_name = fullname.replace(old_path, new_path, 1)
                
                # Show deprecation warning
                warnings.warn(
                    f"Import path '{fullname}' is deprecated. "
                    f"Please use '{new_name}' instead.",
                    DeprecationWarning,
                    stacklevel=2
                )
                
                # Import the new module
                __import__(new_name)
                module = sys.modules[new_name]
                
                # Add to sys.modules with old name for caching
                sys.modules[fullname] = module
                
                return module
        
        raise ImportError(f"No module named '{fullname}'")


def enable_compatibility_mode():
    """
    Enable backward compatibility for old import paths.
    
    This function installs a custom importer that redirects old import paths
    to the new package structure while showing deprecation warnings.
    
    Example:
        >>> from arshai.compat import enable_compatibility_mode
        >>> enable_compatibility_mode()
        >>> 
        >>> # Old imports will now work with warnings
        >>> from arshai.core.interfaces.iagent import IAgent  # Shows deprecation warning
    """
    CompatibilityImporter.install()
    
    # Also create module aliases for direct imports
    import arshai.core.interfaces
    sys.modules['seedwork'] = type(sys)('seedwork')
    sys.modules['seedwork.interfaces'] = arshai.core.interfaces
    
    # Create src module alias
    import arshai
    sys.modules['src'] = arshai
    
    print("Compatibility mode enabled. Old import paths will show deprecation warnings.")


# Convenience imports for backward compatibility
def __getattr__(name):
    """Handle dynamic attribute access for backward compatibility."""
    if name in ['IAgent', 'IAgentConfig', 'IWorkflow', 'ITool']:
        warnings.warn(
            f"Importing '{name}' from arshai.compat is deprecated. "
            f"Please import from arshai.core.interfaces instead.",
            DeprecationWarning,
            stacklevel=2
        )
        from arshai.core.interfaces import __dict__ as interfaces
        return interfaces.get(name)
    raise AttributeError(f"module 'arshai.compat' has no attribute '{name}'")
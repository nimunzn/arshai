"""
Base classes for the Arshai plugin system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Type, Callable
import importlib
import inspect
from pathlib import Path

from arshai.core.interfaces import IAgent, ITool, IWorkflow, IMemoryManager, ILLM


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    author: str
    description: str
    requires: List[str] = None  # List of required dependencies
    tags: List[str] = None  # Tags for categorization
    
    def __post_init__(self):
        if self.requires is None:
            self.requires = []
        if self.tags is None:
            self.tags = []


class Plugin(ABC):
    """
    Base class for all Arshai plugins.
    
    Plugins can extend various aspects of the framework:
    - Add new agent types
    - Add new tools
    - Add new LLM providers
    - Add new memory backends
    - Modify existing behavior through hooks
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the plugin.
        
        Args:
            config: Plugin-specific configuration
        """
        self.config = config or {}
        self._metadata = self.get_metadata()
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the plugin.
        
        This method is called when the plugin is loaded.
        Use it to set up resources, register components, etc.
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """
        Shutdown the plugin.
        
        This method is called when the plugin is unloaded.
        Use it to clean up resources.
        """
        pass
    
    def register_agent(self, name: str, agent_class: Type[IAgent]) -> None:
        """Register a new agent type."""
        from arshai.factories.agent_factory import AgentFactory
        # This would need to be implemented in AgentFactory
        # AgentFactory.register(name, agent_class)
        pass
    
    def register_tool(self, tool: ITool) -> None:
        """Register a new tool."""
        # Tools are typically registered with agents
        pass
    
    def register_llm_provider(self, name: str, llm_class: Type[ILLM]) -> None:
        """Register a new LLM provider."""
        from arshai.factories.llm_factory import LLMFactory
        # This would need to be implemented in LLMFactory
        # LLMFactory.register(name, llm_class)
        pass
    
    def register_memory_provider(self, name: str, memory_class: Type[IMemoryManager]) -> None:
        """Register a new memory provider."""
        from arshai.factories.memory_factory import MemoryFactory
        # This would need to be implemented in MemoryFactory
        # MemoryFactory.register(name, memory_class)
        pass


class PluginRegistry:
    """
    Registry for managing loaded plugins.
    """
    
    def __init__(self):
        self._plugins: Dict[str, Plugin] = {}
        self._metadata: Dict[str, PluginMetadata] = {}
    
    def register(self, plugin: Plugin) -> None:
        """
        Register a plugin.
        
        Args:
            plugin: The plugin instance to register
        """
        metadata = plugin.get_metadata()
        
        if metadata.name in self._plugins:
            raise ValueError(f"Plugin '{metadata.name}' is already registered")
        
        # Check dependencies
        for dep in metadata.requires:
            if dep not in self._plugins:
                raise ValueError(f"Plugin '{metadata.name}' requires '{dep}' which is not loaded")
        
        # Initialize the plugin
        plugin.initialize()
        
        # Store the plugin
        self._plugins[metadata.name] = plugin
        self._metadata[metadata.name] = metadata
    
    def unregister(self, name: str) -> None:
        """
        Unregister a plugin.
        
        Args:
            name: The name of the plugin to unregister
        """
        if name not in self._plugins:
            raise ValueError(f"Plugin '{name}' is not registered")
        
        # Check if other plugins depend on this one
        for other_name, other_meta in self._metadata.items():
            if name in other_meta.requires and other_name != name:
                raise ValueError(f"Cannot unregister '{name}' because '{other_name}' depends on it")
        
        # Shutdown the plugin
        plugin = self._plugins[name]
        plugin.shutdown()
        
        # Remove from registry
        del self._plugins[name]
        del self._metadata[name]
    
    def get(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name."""
        return self._plugins.get(name)
    
    def list_plugins(self) -> List[PluginMetadata]:
        """List all registered plugins."""
        return list(self._metadata.values())
    
    def find_by_tag(self, tag: str) -> List[PluginMetadata]:
        """Find plugins by tag."""
        return [
            meta for meta in self._metadata.values()
            if tag in meta.tags
        ]


# Global plugin registry
_global_registry = PluginRegistry()


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    return _global_registry
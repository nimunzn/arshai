"""
Plugin loader for discovering and loading Arshai plugins.
"""

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import List, Optional, Type, Dict, Any
import json
import yaml

from arshai.extensions.base import Plugin, PluginMetadata, get_plugin_registry


class PluginLoader:
    """
    Loader for discovering and loading plugins.
    
    Plugins can be loaded from:
    - Python packages (installed via pip)
    - Local directories
    - Plugin manifest files
    """
    
    def __init__(self, plugin_paths: Optional[List[Path]] = None):
        """
        Initialize the plugin loader.
        
        Args:
            plugin_paths: Additional paths to search for plugins
        """
        self.plugin_paths = plugin_paths or []
        self.registry = get_plugin_registry()
    
    def discover_plugins(self) -> List[PluginMetadata]:
        """
        Discover available plugins.
        
        Returns:
            List of plugin metadata for discovered plugins
        """
        discovered = []
        
        # Discover from installed packages
        discovered.extend(self._discover_installed_plugins())
        
        # Discover from plugin paths
        for path in self.plugin_paths:
            if path.exists() and path.is_dir():
                discovered.extend(self._discover_directory_plugins(path))
        
        return discovered
    
    def load_plugin(self, name: str, config: Optional[Dict[str, Any]] = None) -> Plugin:
        """
        Load a plugin by name.
        
        Args:
            name: Name of the plugin to load
            config: Configuration for the plugin
            
        Returns:
            Loaded plugin instance
        """
        # Check if already loaded
        existing = self.registry.get(name)
        if existing:
            return existing
        
        # Try to load as installed package
        plugin = self._load_installed_plugin(name, config)
        if plugin:
            self.registry.register(plugin)
            return plugin
        
        # Try to load from plugin paths
        for path in self.plugin_paths:
            plugin = self._load_directory_plugin(path, name, config)
            if plugin:
                self.registry.register(plugin)
                return plugin
        
        raise ValueError(f"Plugin '{name}' not found")
    
    def load_from_manifest(self, manifest_path: Path) -> List[Plugin]:
        """
        Load plugins from a manifest file.
        
        The manifest file can be JSON or YAML format:
        ```yaml
        plugins:
          - name: my_plugin
            config:
              api_key: secret
          - name: another_plugin
        ```
        
        Args:
            manifest_path: Path to the manifest file
            
        Returns:
            List of loaded plugins
        """
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        
        # Load manifest
        with open(manifest_path, 'r') as f:
            if manifest_path.suffix in ['.yaml', '.yml']:
                manifest = yaml.safe_load(f)
            else:
                manifest = json.load(f)
        
        # Load plugins
        loaded = []
        for plugin_spec in manifest.get('plugins', []):
            name = plugin_spec['name']
            config = plugin_spec.get('config', {})
            
            try:
                plugin = self.load_plugin(name, config)
                loaded.append(plugin)
            except Exception as e:
                print(f"Failed to load plugin '{name}': {e}")
                # In production, use proper logging
        
        return loaded
    
    def _discover_installed_plugins(self) -> List[PluginMetadata]:
        """Discover plugins from installed packages."""
        discovered = []
        
        # Look for packages with 'arshai_plugin' entry point
        try:
            import pkg_resources
            
            for entry_point in pkg_resources.iter_entry_points('arshai_plugins'):
                try:
                    plugin_class = entry_point.load()
                    if issubclass(plugin_class, Plugin):
                        # Create temporary instance to get metadata
                        temp_plugin = plugin_class({})
                        discovered.append(temp_plugin.get_metadata())
                except Exception as e:
                    print(f"Error discovering plugin {entry_point.name}: {e}")
        except ImportError:
            # pkg_resources not available
            pass
        
        # Also check for arshai_* packages
        for name, module in sys.modules.items():
            if name.startswith('arshai_') and hasattr(module, 'Plugin'):
                plugin_class = getattr(module, 'Plugin')
                if inspect.isclass(plugin_class) and issubclass(plugin_class, Plugin):
                    try:
                        temp_plugin = plugin_class({})
                        discovered.append(temp_plugin.get_metadata())
                    except Exception:
                        pass
        
        return discovered
    
    def _discover_directory_plugins(self, directory: Path) -> List[PluginMetadata]:
        """Discover plugins from a directory."""
        discovered = []
        
        # Look for plugin.py files
        for plugin_file in directory.rglob('plugin.py'):
            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(
                    f"arshai_plugin_{plugin_file.parent.name}",
                    plugin_file
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find Plugin classes
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, Plugin) and 
                            obj is not Plugin):
                            temp_plugin = obj({})
                            discovered.append(temp_plugin.get_metadata())
            except Exception as e:
                print(f"Error discovering plugin in {plugin_file}: {e}")
        
        return discovered
    
    def _load_installed_plugin(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[Plugin]:
        """Load a plugin from installed packages."""
        # Try direct import
        try:
            module = importlib.import_module(f"arshai_{name}")
            if hasattr(module, 'Plugin'):
                plugin_class = getattr(module, 'Plugin')
                return plugin_class(config or {})
        except ImportError:
            pass
        
        # Try entry points
        try:
            import pkg_resources
            
            for entry_point in pkg_resources.iter_entry_points('arshai_plugins'):
                if entry_point.name == name:
                    plugin_class = entry_point.load()
                    return plugin_class(config or {})
        except ImportError:
            pass
        
        return None
    
    def _load_directory_plugin(
        self,
        directory: Path,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[Plugin]:
        """Load a plugin from a directory."""
        plugin_dir = directory / name
        if not plugin_dir.exists():
            return None
        
        plugin_file = plugin_dir / 'plugin.py'
        if not plugin_file.exists():
            return None
        
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(
                f"arshai_plugin_{name}",
                plugin_file
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                
                # Find Plugin class
                for attr_name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, Plugin) and 
                        obj is not Plugin):
                        return obj(config or {})
        except Exception as e:
            print(f"Error loading plugin {name}: {e}")
        
        return None


# Global plugin loader
_global_loader = PluginLoader()


def get_plugin_loader() -> PluginLoader:
    """Get the global plugin loader."""
    return _global_loader


def load_plugin(name: str, config: Optional[Dict[str, Any]] = None) -> Plugin:
    """
    Convenience function to load a plugin.
    
    Args:
        name: Name of the plugin
        config: Plugin configuration
        
    Returns:
        Loaded plugin instance
    """
    return get_plugin_loader().load_plugin(name, config)
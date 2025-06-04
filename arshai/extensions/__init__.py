"""
Extension system for Arshai framework.

This module provides the plugin architecture for extending Arshai's functionality.
"""

from arshai.extensions.base import Plugin, PluginMetadata, PluginRegistry
from arshai.extensions.hooks import Hook, HookManager
from arshai.extensions.loader import PluginLoader

__all__ = [
    "Plugin",
    "PluginMetadata",
    "PluginRegistry",
    "Hook",
    "HookManager", 
    "PluginLoader",
]
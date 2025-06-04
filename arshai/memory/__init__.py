"""
Memory Management Module

This module provides memory management functionality for AI assistants.
It includes implementations for working memory, short-term memory, and long-term memory.
"""

from .memory_types import ConversationMemoryType
from arshai.core.interfaces.imemorymanager import IMemoryManager, IMemoryInput, IWorkingMemory
# Remove the circular import
# from arshai_src.memory.memory_manager import MemoryManagerService

__all__ = [
    'ConversationMemoryType',
    'IMemoryManager',
    'IMemoryInput',
    'IWorkingMemory',
    # Remove from __all__ as well
    # 'MemoryManagerService',
] 
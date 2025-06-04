"""
Hook system for extending Arshai behavior.
"""

from typing import Callable, Dict, List, Any, Optional
from enum import Enum
import asyncio
from dataclasses import dataclass


class HookType(Enum):
    """Types of hooks available in the system."""
    # Agent hooks
    BEFORE_AGENT_PROCESS = "before_agent_process"
    AFTER_AGENT_PROCESS = "after_agent_process"
    
    # Workflow hooks
    BEFORE_WORKFLOW_START = "before_workflow_start"
    AFTER_WORKFLOW_END = "after_workflow_end"
    BEFORE_NODE_EXECUTE = "before_node_execute"
    AFTER_NODE_EXECUTE = "after_node_execute"
    
    # Memory hooks
    BEFORE_MEMORY_SAVE = "before_memory_save"
    AFTER_MEMORY_SAVE = "after_memory_save"
    BEFORE_MEMORY_RETRIEVE = "before_memory_retrieve"
    AFTER_MEMORY_RETRIEVE = "after_memory_retrieve"
    
    # Tool hooks
    BEFORE_TOOL_EXECUTE = "before_tool_execute"
    AFTER_TOOL_EXECUTE = "after_tool_execute"
    
    # LLM hooks
    BEFORE_LLM_CALL = "before_llm_call"
    AFTER_LLM_CALL = "after_llm_call"


@dataclass
class HookContext:
    """Context passed to hook functions."""
    hook_type: HookType
    data: Dict[str, Any]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Hook:
    """
    Represents a hook that can be registered in the system.
    """
    
    def __init__(
        self,
        name: str,
        hook_type: HookType,
        callback: Callable,
        priority: int = 0,
        enabled: bool = True
    ):
        """
        Initialize a hook.
        
        Args:
            name: Unique name for the hook
            hook_type: Type of hook (when it should be called)
            callback: Function to call when hook is triggered
            priority: Priority for execution order (higher = earlier)
            enabled: Whether the hook is enabled
        """
        self.name = name
        self.hook_type = hook_type
        self.callback = callback
        self.priority = priority
        self.enabled = enabled
    
    async def execute(self, context: HookContext) -> Any:
        """Execute the hook callback."""
        if not self.enabled:
            return None
        
        if asyncio.iscoroutinefunction(self.callback):
            return await self.callback(context)
        else:
            return self.callback(context)


class HookManager:
    """
    Manages hooks for the Arshai framework.
    """
    
    def __init__(self):
        self._hooks: Dict[HookType, List[Hook]] = {
            hook_type: [] for hook_type in HookType
        }
    
    def register_hook(self, hook: Hook) -> None:
        """
        Register a hook.
        
        Args:
            hook: The hook to register
        """
        hooks_list = self._hooks[hook.hook_type]
        
        # Check for duplicate names
        if any(h.name == hook.name for h in hooks_list):
            raise ValueError(f"Hook '{hook.name}' already registered for {hook.hook_type}")
        
        # Add hook and sort by priority
        hooks_list.append(hook)
        hooks_list.sort(key=lambda h: h.priority, reverse=True)
    
    def unregister_hook(self, name: str, hook_type: Optional[HookType] = None) -> None:
        """
        Unregister a hook.
        
        Args:
            name: Name of the hook to unregister
            hook_type: Type of hook (if None, removes from all types)
        """
        if hook_type:
            self._hooks[hook_type] = [
                h for h in self._hooks[hook_type] if h.name != name
            ]
        else:
            for hook_list in self._hooks.values():
                hook_list[:] = [h for h in hook_list if h.name != name]
    
    async def execute_hooks(
        self,
        hook_type: HookType,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Execute all hooks of a given type.
        
        Args:
            hook_type: Type of hooks to execute
            data: Data to pass to hooks
            metadata: Additional metadata
            
        Returns:
            List of results from hook executions
        """
        context = HookContext(
            hook_type=hook_type,
            data=data,
            metadata=metadata or {}
        )
        
        results = []
        for hook in self._hooks[hook_type]:
            if hook.enabled:
                try:
                    result = await hook.execute(context)
                    results.append(result)
                    
                    # Allow hooks to modify context data
                    if isinstance(result, dict) and "modified_data" in result:
                        context.data.update(result["modified_data"])
                        
                except Exception as e:
                    # Log error but don't stop execution
                    print(f"Error in hook '{hook.name}': {e}")
                    # In production, use proper logging
        
        return results
    
    def get_hooks(self, hook_type: HookType) -> List[Hook]:
        """Get all hooks of a given type."""
        return self._hooks[hook_type].copy()
    
    def enable_hook(self, name: str) -> None:
        """Enable a hook by name."""
        for hook_list in self._hooks.values():
            for hook in hook_list:
                if hook.name == name:
                    hook.enabled = True
    
    def disable_hook(self, name: str) -> None:
        """Disable a hook by name."""
        for hook_list in self._hooks.values():
            for hook in hook_list:
                if hook.name == name:
                    hook.enabled = False


# Global hook manager
_global_hook_manager = HookManager()


def get_hook_manager() -> HookManager:
    """Get the global hook manager."""
    return _global_hook_manager


# Decorator for easy hook registration
def hook(hook_type: HookType, name: Optional[str] = None, priority: int = 0):
    """
    Decorator for registering a function as a hook.
    
    Example:
        @hook(HookType.BEFORE_AGENT_PROCESS, priority=10)
        def my_hook(context: HookContext):
            print(f"Processing: {context.data}")
    """
    def decorator(func: Callable):
        hook_name = name or func.__name__
        hook_instance = Hook(
            name=hook_name,
            hook_type=hook_type,
            callback=func,
            priority=priority
        )
        get_hook_manager().register_hook(hook_instance)
        return func
    
    return decorator
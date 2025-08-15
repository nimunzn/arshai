"""
DEPRECATED: Integration with LLM factory for automatic observability.

⚠️  DEPRECATION WARNING ⚠️

This factory-based approach is DEPRECATED and will be removed in a future version.

The new constructor-based approach is much simpler and cleaner:

    from arshai.llms.openai import OpenAIClient
    from arshai.observability import ObservabilityManager, ObservabilityConfig
    
    # Create observability manager
    obs_config = ObservabilityConfig(service_name="my-app")
    obs_manager = ObservabilityManager(obs_config)
    
    # Use client constructor directly - no factory wrapper needed!
    client = OpenAIClient(config, observability_manager=obs_manager)

This is cleaner, more direct, and eliminates the complexity of factory wrappers.

MIGRATION PATH:
- Replace ObservableFactory with direct client constructors
- Replace create_observable_factory() with ObservabilityManager + client constructor
- Use client.chat() instead of client.chat_completion()

This module will be removed in the next major version.
"""

import logging
from typing import Dict, Any, Type, Optional, Union
from pathlib import Path

from arshai.core.interfaces import ILLM, ILLMConfig
from .core import ObservabilityManager
from .config import ObservabilityConfig
from .decorators import with_observability, ObservabilityMixin


class ObservableFactory:
    """Factory wrapper that adds observability to LLM providers."""
    
    def __init__(self, 
                 base_factory_class,
                 observability_config: Optional[ObservabilityConfig] = None,
                 config_path: Optional[Union[str, Path]] = None):
        """Initialize observable factory.
        
        Args:
            base_factory_class: The base LLM factory class to wrap
            observability_config: Optional observability configuration
            config_path: Optional path to configuration file
        """
        self.base_factory = base_factory_class
        self.logger = logging.getLogger(__name__)
        
        # Load observability configuration
        if observability_config:
            self.observability_config = observability_config
        elif config_path:
            self.observability_config = ObservabilityConfig.from_yaml(config_path)
        else:
            self.observability_config = ObservabilityConfig.from_config_file_or_env()
        
        # Create observability manager
        self.observability_manager = ObservabilityManager(self.observability_config)
        
        # Provider registry (copy from base factory)
        self._providers = getattr(base_factory_class, '_providers', {}).copy()
        
        self.logger.info("Observable factory initialized")
    
    def create(self,
               provider: str,
               config: ILLMConfig,
               **kwargs) -> ILLM:
        """Create an LLM provider instance with observability.
        
        Args:
            provider: The provider type (e.g., 'openai', 'azure')
            config: Configuration for the LLM
            **kwargs: Additional configuration parameters
            
        Returns:
            An instrumented instance of the specified LLM provider
        """
        # Create the original LLM instance using the base factory
        original_instance = self.base_factory.create(provider, config, **kwargs)
        
        # Check if observability should be added for this provider
        if not self.observability_config.is_token_timing_enabled(provider):
            self.logger.debug(f"Observability disabled for provider: {provider}")
            return original_instance
        
        # Add observability to the instance
        return self._add_observability(original_instance, provider)
    
    def _add_observability(self, instance: ILLM, provider: str) -> ILLM:
        """Add observability to an LLM instance.
        
        Args:
            instance: LLM instance to instrument
            provider: Provider name
            
        Returns:
            Instrumented LLM instance
        """
        if self.observability_config.non_intrusive:
            # Non-intrusive mode: wrap methods without modifying the instance
            return self._wrap_instance_methods(instance, provider)
        else:
            # Intrusive mode: modify the instance directly
            return self._modify_instance_methods(instance, provider)
    
    def _wrap_instance_methods(self, instance: ILLM, provider: str) -> ILLM:
        """Wrap instance methods with observability (non-intrusive).
        
        Args:
            instance: LLM instance
            provider: Provider name
            
        Returns:
            Wrapped instance
        """
        # Create a wrapper class dynamically
        class ObservableWrapper:
            def __init__(self, wrapped_instance, provider_name, obs_manager):
                self._wrapped = wrapped_instance
                self._provider = provider_name
                self._observability_manager = obs_manager
                
                # Copy all attributes from the wrapped instance
                for attr_name in dir(wrapped_instance):
                    if not attr_name.startswith('_'):
                        attr_value = getattr(wrapped_instance, attr_name)
                        if not callable(attr_value):
                            setattr(self, attr_name, attr_value)
            
            def __getattr__(self, name):
                # If it's one of the LLM methods, create a wrapper
                if name in ['chat_completion', 'chat_with_tools', 'stream_completion', 'stream_with_tools']:
                    # Get the original method
                    original_method = getattr(self._wrapped, name)
                    
                    # Create a wrapper that properly handles the bound method
                    def method_wrapper(llm_input, *args, **kwargs):
                        # Create a temporary wrapper class that has the required attributes
                        class TempWrapper:
                            def __init__(self, wrapped_instance):
                                self.config = wrapped_instance.config if hasattr(wrapped_instance, 'config') else None
                        
                        temp_instance = TempWrapper(self._wrapped)
                        
                        # Create a properly named function instead of lambda
                        def observable_func(self, llm_input, *a, **kw):
                            return original_method(llm_input, *a, **kw)
                        
                        # Set the function name to match the original method
                        observable_func.__name__ = name
                        
                        # Create the observable version
                        observable_method = with_observability(
                            self._provider, 
                            self._observability_manager
                        )(observable_func)
                        
                        # Call it with the temp instance
                        return observable_method(temp_instance, llm_input, *args, **kwargs)
                    
                    return method_wrapper
                
                # For other attributes, return as normal
                return getattr(self._wrapped, name)
        
        return ObservableWrapper(instance, provider, self.observability_manager)
    
    def _modify_instance_methods(self, instance: ILLM, provider: str) -> ILLM:
        """Modify instance methods directly (intrusive).
        
        Args:
            instance: LLM instance
            provider: Provider name
            
        Returns:
            Modified instance
        """
        # Add observability manager to the instance
        instance._observability_manager = self.observability_manager
        
        # List of methods to make observable
        methods_to_observe = [
            'chat_completion', 
            'chat_with_tools', 
            'stream_completion', 
            'stream_with_tools'
        ]
        
        for method_name in methods_to_observe:
            if hasattr(instance, method_name):
                original_method = getattr(instance, method_name)
                observable_method = with_observability(
                    provider, 
                    self.observability_manager
                )(original_method)
                setattr(instance, method_name, observable_method)
        
        return instance
    
    @classmethod
    def register(cls, name: str, provider_class: Type[ILLM]) -> None:
        """Register a new LLM provider (delegates to base factory).
        
        Args:
            name: Name of the provider
            provider_class: Class implementing the ILLM interface
        """
        # This would need to be implemented based on the base factory
        pass
    
    def get_observability_manager(self) -> ObservabilityManager:
        """Get the observability manager instance."""
        return self.observability_manager
    
    def get_supported_providers(self) -> list:
        """Get list of supported providers."""
        return list(self._providers.keys())
    
    def is_observability_enabled(self, provider: str = None) -> bool:
        """Check if observability is enabled.
        
        Args:
            provider: Optional provider name to check
            
        Returns:
            True if observability is enabled
        """
        if provider:
            return self.observability_config.is_token_timing_enabled(provider)
        return True  # Always enabled now


def create_observable_factory(base_factory_class,
                            observability_config: Optional[ObservabilityConfig] = None,
                            config_path: Optional[Union[str, Path]] = None) -> ObservableFactory:
    """Create an observable factory from a base factory class.
    
    Args:
        base_factory_class: The base LLM factory class
        observability_config: Optional observability configuration
        config_path: Optional path to configuration file
        
    Returns:
        ObservableFactory instance
    """
    return ObservableFactory(base_factory_class, observability_config, config_path)


# Factory integration function for easy usage
def make_factory_observable(factory_instance, 
                           observability_config: Optional[ObservabilityConfig] = None) -> ObservableFactory:
    """Make an existing factory instance observable.
    
    Args:
        factory_instance: Existing factory instance
        observability_config: Optional observability configuration
        
    Returns:
        ObservableFactory wrapping the original factory
    """
    return ObservableFactory(factory_instance.__class__, observability_config)
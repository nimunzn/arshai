"""
Example demonstrating how to use the Arshai plugin system.
"""

import asyncio
from pathlib import Path

from arshai import Settings
from arshai.extensions import PluginLoader, get_plugin_registry, get_hook_manager
from arshai.extensions.hooks import HookType
from arshai.core.interfaces import IAgentConfig, IAgentInput


async def main():
    """Main function demonstrating plugin usage."""
    
    print("=== Arshai Plugin System Example ===\n")
    
    # 1. Initialize plugin loader with custom plugin path
    plugin_paths = [Path("examples")]  # Look for plugins in examples directory
    loader = PluginLoader(plugin_paths=plugin_paths)
    
    # 2. Discover available plugins
    print("Discovering plugins...")
    discovered = loader.discover_plugins()
    for meta in discovered:
        print(f"  Found: {meta.name} v{meta.version} - {meta.description}")
    
    # 3. Load the example plugin with configuration
    print("\nLoading example plugin...")
    plugin_config = {
        "verbose": True,
        "log_inputs": True
    }
    example_plugin = loader.load_plugin("example_plugin", config=plugin_config)
    
    # 4. List loaded plugins
    print("\nLoaded plugins:")
    registry = get_plugin_registry()
    for plugin_meta in registry.list_plugins():
        print(f"  - {plugin_meta.name} ({', '.join(plugin_meta.tags)})")
    
    # 5. Use the plugin's custom tool
    print("\nUsing plugin's word count tool:")
    word_count_tool = example_plugin.get_word_count_tool()
    result = await word_count_tool.execute(
        text="This is a test sentence with eight words."
    )
    print(f"  Result: {result}")
    
    # 6. Demonstrate hooks in action
    print("\nDemonstrating hooks...")
    hook_manager = get_hook_manager()
    
    # Simulate agent processing with hooks
    agent_input = IAgentInput(
        message="Hello, how can you help me?",
        conversation_id="test_123"
    )
    
    # Execute before hooks
    print("  Executing BEFORE_AGENT_PROCESS hooks...")
    await hook_manager.execute_hooks(
        HookType.BEFORE_AGENT_PROCESS,
        data={"input": agent_input}
    )
    
    # Simulate agent response
    response = "I'm here to help you with any questions!"
    
    # Execute after hooks
    print("  Executing AFTER_AGENT_PROCESS hooks...")
    results = await hook_manager.execute_hooks(
        HookType.AFTER_AGENT_PROCESS,
        data={"response": response}
    )
    
    # Check if response was modified by hooks
    for result in results:
        if isinstance(result, dict) and "modified_data" in result:
            print(f"  Response modified by hook: {result}")
    
    # 7. Demonstrate plugin manifest loading
    print("\nCreating plugin manifest...")
    manifest_path = Path("plugin_manifest.yaml")
    manifest_content = """
plugins:
  - name: example_plugin
    config:
      verbose: false
      log_inputs: true
"""
    
    with open(manifest_path, 'w') as f:
        f.write(manifest_content)
    
    print(f"  Created manifest: {manifest_path}")
    
    # Clean up
    manifest_path.unlink()
    
    # 8. Unload plugin
    print("\nUnloading plugin...")
    registry.unregister("example_plugin")
    print("  Plugin unloaded")
    
    print("\n=== Example Complete ===")


# Example of creating a custom plugin in user code
from arshai.extensions.base import Plugin, PluginMetadata


class CustomUserPlugin(Plugin):
    """Example of a user-defined plugin."""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="custom_user_plugin",
            version="0.1.0",
            author="User",
            description="Custom plugin created in user code",
            tags=["custom", "user"]
        )
    
    def initialize(self) -> None:
        print(f"Custom plugin initialized with config: {self.config}")
    
    def shutdown(self) -> None:
        print("Custom plugin shutdown")


if __name__ == "__main__":
    asyncio.run(main())
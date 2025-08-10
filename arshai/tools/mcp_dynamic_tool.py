"""
Dynamic MCP Tool Wrapper

Creates individual ITool instances for each MCP server tool, following the proper
MCP pattern where each server tool becomes a distinct client-side tool.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

from arshai.core.interfaces.itool import ITool
from arshai.clients.mcp.server_manager import MCPServerManager
from arshai.clients.mcp.exceptions import MCPError, MCPConnectionError, MCPToolError

logger = logging.getLogger(__name__)


class MCPDynamicTool(ITool):
    """
    Individual ITool wrapper for a specific MCP server tool.
    
    This class creates a dedicated ITool instance for each tool discovered
    from MCP servers, following the MCP pattern of individual tool definitions.
    """
    
    # Class-level shared thread pool
    _executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
    _executor_lock = threading.Lock()
    
    @classmethod
    def _get_executor(cls) -> concurrent.futures.ThreadPoolExecutor:
        """Get or create the shared thread pool executor."""
        if cls._executor is None:
            with cls._executor_lock:
                if cls._executor is None:
                    # Calculate optimal thread count
                    max_workers = min(
                        int(os.getenv("ARSHAI_MAX_THREADS", "32")),
                        (os.cpu_count() or 1) * 2
                    )
                    cls._executor = concurrent.futures.ThreadPoolExecutor(
                        max_workers=max_workers,
                        thread_name_prefix="mcp_tool"
                    )
                    logger.info(f"Created MCP tool thread pool with {max_workers} workers")
        return cls._executor
    
    def __init__(self, tool_spec: Dict[str, Any], server_manager: MCPServerManager):
        """
        Initialize the MCP dynamic tool.
        
        Args:
            tool_spec: Tool specification from MCP server discovery
            server_manager: Manager for handling MCP server connections
        """
        self.tool_spec = tool_spec
        self.server_manager = server_manager
        
        # Extract tool information
        self.name = tool_spec['name']
        self.description = tool_spec['description']
        self.server_name = tool_spec['server_name']
        self.server_url = tool_spec.get('server_url', '')
        self._input_schema = tool_spec.get('inputSchema', {})
        
        # Create unique tool name to avoid conflicts across servers
        self.unique_name = f"{self.server_name}_{self.name}"
        
    @property
    def function_definition(self) -> Dict[str, Any]:
        """
        Convert MCP tool spec to OpenAI function format.
        
        Returns:
            Function definition in OpenAI format for the LLM
        """
        # Use the original tool name for the function (not unique name)
        # The LLM should see the original tool names from the MCP server
        function_def = {
            "name": self.name,
            "description": f"{self.description} (Server: {self.server_name})",
            "parameters": self._input_schema or {
                "type": "object",
                "properties": {},
                "additionalProperties": True
            }
        }
        
        return function_def
    
    def execute(self, **kwargs) -> Any:
        """
        Synchronous execution of the MCP tool.
        
        Args:
            **kwargs: Tool arguments
            
        Returns:
            Tool execution result formatted for LLM consumption
        """
        try:
            # Try to determine if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an event loop, use shared thread pool
                executor = self._get_executor()
                future = executor.submit(self._run_sync_in_thread, kwargs)
                result = future.result(timeout=60)  # 60 second timeout
                return "function", result
                    
            except RuntimeError:
                # No event loop running, safe to use asyncio.run()
                return "function", asyncio.run(self._execute_async(**kwargs))
                
        except concurrent.futures.TimeoutError:
            error_msg = f"Tool '{self.name}' on server '{self.server_name}' timed out after 60 seconds"
            logger.error(error_msg)
            return "function", [{"type": "text", "text": error_msg}]
        except Exception as e:
            error_msg = f"Error executing MCP tool '{self.name}' on server '{self.server_name}': {e}"
            logger.error(error_msg)
            return "function", [{"type": "text", "text": error_msg}]
    
    def _run_sync_in_thread(self, kwargs):
        """Run async method in thread with new event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._execute_async(**kwargs))
        finally:
            loop.close()
    
    async def aexecute(self, **kwargs) -> Any:
        """
        Asynchronous execution of the MCP tool.
        
        Args:
            **kwargs: Tool arguments
            
        Returns:
            Tool execution result formatted for LLM consumption
        """
        try:
            return "function", await self._execute_async(**kwargs)
        except Exception as e:
            error_msg = f"Error executing MCP tool '{self.name}' on server '{self.server_name}': {e}"
            logger.error(error_msg)
            return "function", [{"type": "text", "text": error_msg}]
    
    async def _execute_async(self, **kwargs) -> Any:
        """
        Internal async method to execute the MCP tool.
        
        Args:
            **kwargs: Tool arguments
            
        Returns:
            Tool execution result formatted for LLM consumption
        """
        try:
            # Create a fresh MCP client connection for this tool execution
            from arshai.clients.mcp.base_client import BaseMCPClient
            from arshai.clients.mcp.config import MCPServerConfig
            
            logger.info(f"Executing MCP tool '{self.name}' on server '{self.server_name}' with arguments: {kwargs}")
            
            # Create server config and client
            server_config = MCPServerConfig(
                name=self.server_name,
                url=self.server_url,
                timeout=30,
                max_retries=3
            )
            client = BaseMCPClient(server_config)
            
            try:
                # Connect to the server
                await client.connect()
                
                # Execute the tool
                result = await client.call_tool(self.name, kwargs)
                
                # Format result for LLM consumption
                formatted_result = self._format_result_for_llm(result)
                
                logger.info(f"MCP tool '{self.name}' on server '{self.server_name}' executed successfully")
                return formatted_result
                
            finally:
                # Always disconnect the client
                try:
                    await client.disconnect()
                except Exception as disconnect_error:
                    logger.debug(f"Minor cleanup issue when disconnecting from '{self.server_name}': {disconnect_error}")
                
        except MCPConnectionError as e:
            error_msg = f"Connection error when executing '{self.name}' on server '{self.server_name}': {str(e)}"
            logger.error(error_msg)
            return [{"type": "text", "text": error_msg}]
            
        except MCPToolError as e:
            error_msg = f"Tool execution error for '{self.name}' on server '{self.server_name}': {str(e)}"
            logger.error(error_msg)
            
            # Check if it's a tool not found error and provide helpful message
            if "not found" in str(e).lower() or "unknown" in str(e).lower():
                error_msg += f"\n\nTip: Make sure the tool '{self.name}' is still available on server '{self.server_name}'"
            
            return [{"type": "text", "text": error_msg}]
            
        except MCPError as e:
            error_msg = f"MCP error when executing '{self.name}' on server '{self.server_name}': {str(e)}"
            logger.error(error_msg)
            return [{"type": "text", "text": error_msg}]
            
        except Exception as e:
            error_msg = f"Unexpected error when executing '{self.name}' on server '{self.server_name}': {str(e)}"
            logger.error(error_msg)
            return [{"type": "text", "text": error_msg}]
    
    def _format_result_for_llm(self, result: Any) -> List[Dict[str, str]]:
        """
        Format MCP tool result for LLM consumption.
        
        Args:
            result: Raw result from MCP tool execution
            
        Returns:
            Formatted result for LLM
        """
        try:
            if isinstance(result, dict):
                result_text = json.dumps(result, indent=2, ensure_ascii=False)
            elif isinstance(result, (list, tuple)):
                result_text = json.dumps(result, indent=2, ensure_ascii=False)
            else:
                result_text = str(result)
            
            response_text = f"Tool '{self.name}' executed successfully on server '{self.server_name}'.\n\nResult:\n{result_text}"
            return [{"type": "text", "text": response_text}]
            
        except Exception as e:
            logger.warning(f"Failed to format result for tool '{self.name}': {e}")
            # Fallback to simple string conversion
            response_text = f"Tool '{self.name}' executed successfully on server '{self.server_name}'.\n\nResult:\n{str(result)}"
            return [{"type": "text", "text": response_text}]
    
    def __repr__(self) -> str:
        """String representation of the tool."""
        return f"MCPDynamicTool(name='{self.name}', server='{self.server_name}', url='{self.server_url}')"
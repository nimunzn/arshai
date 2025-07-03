"""
MCP-specific exceptions for better error handling and categorization.
"""


class MCPError(Exception):
    """Base exception for all MCP-related errors."""
    
    def __init__(self, message: str, server_name: str = None, original_error: Exception = None):
        super().__init__(message)
        self.message = message
        self.server_name = server_name
        self.original_error = original_error
    
    def __str__(self):
        base_msg = self.message
        if self.server_name:
            base_msg = f"[{self.server_name}] {base_msg}"
        if self.original_error:
            base_msg += f" (caused by: {self.original_error})"
        return base_msg


class MCPConnectionError(MCPError):
    """Exception raised when connection to MCP server fails."""
    pass


class MCPToolError(MCPError):
    """Exception raised when MCP tool execution fails."""
    
    def __init__(self, message: str, tool_name: str = None, server_name: str = None, original_error: Exception = None):
        super().__init__(message, server_name, original_error)
        self.tool_name = tool_name
    
    def __str__(self):
        base_msg = self.message
        if self.tool_name:
            base_msg = f"Tool '{self.tool_name}': {base_msg}"
        if self.server_name:
            base_msg = f"[{self.server_name}] {base_msg}"
        if self.original_error:
            base_msg += f" (caused by: {self.original_error})"
        return base_msg


class MCPConfigurationError(MCPError):
    """Exception raised when MCP configuration is invalid."""
    pass


class MCPTimeoutError(MCPError):
    """Exception raised when MCP operation times out."""
    pass
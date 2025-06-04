"""Memory type definitions."""

from enum import Enum, auto
from typing import Literal

class ConversationMemoryType(str, Enum):
    """Enumeration of memory types supported by the system."""
    
    SHORT_TERM_MEMORY = "SHORT_TERM_MEMORY"
    LONG_TERM_MEMORY = "LONG_TERM_MEMORY"
    WORKING_MEMORY = "WORKING_MEMORY"
    
    def __str__(self) -> str:
        return self.value 
from typing import Protocol, Optional, List, Dict, Any, Union
from pydantic import Field
from .idto import IDTO, IStreamDTO
from enum import StrEnum

class ConversationMemoryType(StrEnum):
    SHORT_TERM_MEMORY = "SHORT_TERM_MEMORY"
    LONG_TERM_MEMORY = "LONG_TERM_MEMORY"
    WORKING_MEMORY = "WORKING_MEMORY"

class IMemoryConfig(IDTO):
    """
    Configuration for memory systems.
    
    This class represents configuration settings for different memory types
    such as working memory, short-term memory, and long-term memory.
    """
    working_memory: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for working memory"
    )
    short_term_memory: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for short-term memory"
    )
    long_term_memory: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for long-term memory"
    )

class IMemoryItem(IDTO):
    """
    Represents a single item in conversation memory.
    
    This class represents a message or other item stored in conversation memory,
    with metadata about the item.
    """
    role: str = Field(
        description="Role of the message sender (e.g., 'user', 'assistant', 'system')"
    )
    content: str = Field(
        description="Content of the memory item"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata for the memory item"
    )

class IWorkingMemory(IDTO):
    """
    Maintains the assistant's working memory during conversations.
    
    Contains a single structured string field that encompasses all Working Memory components
    in a format optimized for LLM understanding and processing.
    """
    
    working_memory: str = Field(
        description="""
        A structured string containing all Working Memory components.

        Each section should be clearly delineated and formatted in a way that's natural for LLM processing.
        Updates should maintain this structure while updating relevant sections.
        """
    )

    @classmethod
    def initialize_memory(cls) -> 'IWorkingMemory':
        """Create a new working memory state with initial values"""
        initial_working_memory = """
        USER PROFILE:
        New user with no established profile yet.

        AGENT PROFILE:
        General chatbot assistant awaiting specific role definition.

        CONVERSATION STORY:
        Conversation just initiated.

        CURRENT CONVERSATION HISTORY:
        No relevant interaction yet.

        DIALOGUE PLANNING AND GOALS:
        no goals yet.

        KNOWLEDGE CACHE:
        No cached knowledge yet.

        CONVERSATION MOOD:
        Neutral
        """
        return cls(working_memory=initial_working_memory.strip())
    
    def to_dict(self) -> Dict:
        """Convert the working memory state to dictionary format for storage"""
        return {
            "working_memory": self.working_memory
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'IWorkingMemory':
        """Create working memory state from stored dictionary format"""
        return cls(working_memory=data["working_memory"])


class IStreamWorkingMemory(IStreamDTO):
    """
    Maintains the assistant's working memory during conversations.
    
    Contains a single structured string field that encompasses all Working Memory components
    in a format optimized for LLM understanding and processing.
    """
    
    working_memory: str = Field(
        description="""
        A structured string containing all Working Memory components.

        Each section should be clearly delineated and formatted in a way that's natural for LLM processing.
        Updates should maintain this structure while updating relevant sections.
        """
    )

    @classmethod
    def initialize_memory(cls) -> 'IStreamWorkingMemory':
        """Create a new working memory state with initial values"""
        initial_working_memory = """
        USER PROFILE:
        New user with no established profile yet.

        AGENT PROFILE:
        General chatbot assistant awaiting specific role definition.

        CONVERSATION STORY:
        Conversation just initiated.

        CURRENT CONVERSATION HISTORY:
        No relevant interaction yet.

        DIALOGUE PLANNING AND GOALS:
        no goals yet.

        KNOWLEDGE CACHE:
        No cached knowledge yet.

        CONVERSATION MOOD:
        Neutral
        """
        return cls(working_memory=initial_working_memory.strip())
    
    def to_dict(self) -> Dict:
        """Convert the working memory state to dictionary format for storage"""
        return {
            "working_memory": self.working_memory
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'IStreamWorkingMemory':
        """Create working memory state from stored dictionary format"""
        return cls(working_memory=data["working_memory"])

class IMemoryInput(IDTO):
    """
    Represents the input for memory operations.
    
    Attributes:
        conversation_id: Unique identifier for the conversation
        memory_type: Type of memory to operate on
        data: Memory data for store/update operations
        query: Search query for retrieve operations
        memory_id: Specific memory ID for update/delete operations
        limit: Maximum number of items to retrieve
        filters: Additional filters to apply
        metadata: Additional metadata for the memory entry
    """
    conversation_id: str = Field(description="Unique identifier for the conversation")
    memory_type: ConversationMemoryType = Field(
        default=ConversationMemoryType.SHORT_TERM_MEMORY,
        description="Type of memory to operate on"
    )
    data: Optional[List[IWorkingMemory]] = Field(
        default=None,
        description="Memory data for store/update operations"
    )
    query: Optional[str] = Field(
        default=None,
        description="Search query for retrieve operations"
    )
    memory_id: Optional[str] = Field(
        default=None,
        description="Specific memory ID for update/delete operations"
    )
    limit: Optional[int] = Field(
        default=5,
        description="Maximum number of items to retrieve"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional filters to apply"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for the memory entry"
    )


class IMemoryManager(Protocol):
    """
    Universal interface for memory management systems.
    Provides a unified approach to handling different types of memory:
    - Short-term memory: Recent conversations and temporary information
    - Long-term memory: Persistent knowledge and important information
    - Context management: Current conversation state and context
    
    Implementations can choose to implement all or specific memory management features.
    """
    
    def store(self, input: IMemoryInput) -> str:
        """
        Universal method to store any type of memory data.

        Args:
            input: IMemoryInput containing data and metadata for storage

        Returns:
            str: Unique identifier for the stored memory
        """
        ...

    def retrieve(self, input: IMemoryInput) -> List[IWorkingMemory]:
        """
        Universal method to retrieve any type of memory data.

        Args:
            input: IMemoryInput containing query and retrieval parameters

        Returns:
            List[IWorkingMemory]: Matching memory entries
        """
        ...

    def update(self, input: IMemoryInput) -> None:
        """
        Universal method to update any type of memory data.

        Args:
            input: IMemoryInput containing memory ID and update data
        """
        ...

    def delete(self, input: IMemoryInput) -> None:
        """
        Universal method to delete memory data.

        Args:
            input: IMemoryInput containing memory ID or deletion criteria
        """
        ...


import json
import os
import logging
from typing import Dict, List, Optional, Any
from uuid import uuid4
from datetime import datetime
import aiohttp
from urllib.parse import quote

logger = logging.getLogger(__name__)

class ChatHistoryClient:
    """Client for interacting with Chat History Service including forum functionality."""
    
    def __init__(self, header_data: Dict):
        """Initialize the Chat History client.
        
        Args:
            settings: Settings containing API configuration
        """
        self.api_url = os.getenv("FAST_CHAT_URL", "http://localhost:8000")
        self.header_data = header_data

    async def _request(self, method: str, endpoint: str, data: Optional[Dict] = None, headers: Any = None) -> Dict:
        """Make a request to the Chat History API.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE, PATCH)
            endpoint: API endpoint path
            data: Request data for POST/PUT/PATCH requests
            
        Returns:
            Response data
        """
        url = f"{self.api_url}{endpoint}"
        
        headers = {
            "Content-Type": "application/json",
            "X-Auth-User": json.dumps(self.header_data)  # Format as JSON with user_id field
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                logger.debug(f"url: {url}")
                logger.debug(f"headers: {headers}")
                logger.debug(f"data: {data}")

                if method == "GET":
                    async with session.get(url, headers=headers) as response:
                        response.raise_for_status()
                        return await response.json()
                        
                elif method == "POST":
                    async with session.post(url, headers=headers, json=data) as response:
                        response.raise_for_status()
                        return await response.json()
                        
                elif method == "PUT":
                    async with session.put(url, headers=headers, json=data) as response:
                        response.raise_for_status()
                        return await response.json()
                        
                elif method == "DELETE":
                    async with session.delete(url, headers=headers) as response:
                        response.raise_for_status()
                        return await response.json()
                
                elif method == "PATCH":
                    async with session.patch(url, headers=headers, json=data) as response:
                        response.raise_for_status()
                        return await response.json()
                        
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                    
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error in Chat History API request: {str(e)}")
            raise
            
        except Exception as e:
            logger.error(f"Error in Chat History API request: {str(e)}")
            raise
    
    async def create_conversation(self, **kwargs) -> str:
        """Create a new conversation.
        
        Args:
            **kwargs: Optional fields for the conversation including:
                correlation_id: Correlation ID for tracking
                org: Organization ID
                first_name: User's first name
                last_name: User's last name
                conversation_id: Optional ID for the conversation
                state: Conversation state
                messages: List of messages to include
                realm: Realm identifier
                is_anonymous: Whether the user is anonymous
                agent_title: Title of the agent
            
        Returns:
            Conversation ID
        """
        created_at = datetime.now().isoformat()
        state = kwargs.get("state", "normal")
        
        # If messages are provided, use them instead of default
        messages = kwargs.pop("messages", [])
        
        data = {
            "version": 0,
            "first_name": kwargs.get("first_name", "Name"),
            "last_name": kwargs.get("last_name", "Family"),
            "state": state,  # "normal","finish","archive","human-intervention"
            "messages": messages,
            "model": kwargs.get("model"),
            "temperature": kwargs.get("temperature",0.0),
            "title": kwargs.get("title", "مکالمه جدید"),
            "org": kwargs.get("org"),
            "user": kwargs.get("user"),
            "realm": kwargs.get("realm"),
            "agent_title": kwargs.get("agent_title", ""),
            "is_anonymous": kwargs.get("is_anonymous", True),
            "created_at": created_at,
            "updated_at": created_at,
        }
        
        # Add any remaining kwargs
        data.update(kwargs)
        

        response = await self._request("POST", "/user/conversations", data)
        return response.get("data", {}).get("conversation_id")

    
    async def get_conversation(self, conversation_id: str) -> Dict:
        """Get conversation details.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Conversation details
        """
        response = await self._request("GET", f"/user/conversations/{conversation_id}")
        return response.get("data", {})
    
    async def get_conversation_state(self, conversation_id: str) -> str:
        """Get the state of a conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Conversation state (normal, finish, archive, human-intervention, etc.)
        """
        response = await self._request("GET", f"/admin/conversation-state/{conversation_id}")
        state = response.get("data", {}).get("state")
        logger.debug(f"State for conversation {conversation_id} is: {state}")
        return state
    
    async def check_conversation_id(self, conversation_id: str) -> bool:
        """Check if a conversation ID is valid and accessible to the current user.
        
        Args:
            conversation_id: ID of the conversation to check
            
        Returns:
            Boolean indicating whether the conversation exists and is accessible
        """
        try:
            response = await self._request("GET", f"/check-conversation-id?conversation_id={conversation_id}")
            return response.get("success", False)
        except Exception as e:
            logger.error(f"Error checking conversation ID: {str(e)}")
            return False
    
    async def get_messages(self, conversation_id: str, page: int = 1, page_size: int = 10) -> Dict:
        """Get all messages for a conversation with pagination.
        
        Args:
            conversation_id: ID of the conversation
            page: Page number (default: 1)
            page_size: Number of items per page (default: 10)
            
        Returns:
            Dictionary containing messages list with pagination info
        """
        endpoint = f"/user/conversations/{conversation_id}/messages?page={page}&pageSize={page_size}"
        response = await self._request("GET", endpoint)
        return response.get("data", {})

    async def get_messages_by_metadata(
        self, 
        conversation_id: str, 
        metadata: List[str],
        page: int = 1, 
        page_size: int = 10
    ) -> Dict:
        """Get messages for a conversation filtered by metadata with pagination.
        
        Args:
            conversation_id: ID of the conversation
            metadata: List of metadata strings to filter by
            page: Page number (default: 1)
            page_size: Number of items per page (default: 10, max: 1000)
            
        Returns:
            Dictionary containing filtered messages with pagination info
        """
        # Build the endpoint URL with base query parameters
        endpoint = f"/user/conversations/{conversation_id}/messages?page={page}&pageSize={page_size}"
        
        # Add metadata parameters - the API expects multiple metadata parameters
        # Example: &metadata=tag1&metadata=tag2
        if metadata:
            for meta in metadata:
                endpoint += f"&metadata={quote(meta)}"
            
        response = await self._request("GET", endpoint)
        return response.get("data", {})
    
    async def update_message(self, message_id: str, updates: Dict) -> Dict:
        """Update a message.
        
        Args:
            message_id: ID of the message to update
            updates: Fields to update
            
        Returns:
            Updated message
        """
        # Note: This endpoint wasn't clearly specified in the new API documentation
        # Using the old endpoint for now
        response = await self._request("PATCH", f"/messages/{message_id}", updates)
        return response.get("data", {})
    
    async def add_message(self, data: Dict) -> Dict:
        """Add a message to an existing conversation.
        
        Args:
            conversation_id: ID of the conversation
            message: Message data to add
            
        Returns:
            Added message details
        """
        response = await self._request("POST", f"/user/conversations", data)
        return response.get("data", {})
    
    async def update_conversation_state(self, conversation_id: str, state: str) -> Dict:
        """Update the state of a conversation.
        
        Args:
            conversation_id: ID of the conversation
            state: New state for the conversation ('normal', 'finish', 'archive', 'human-intervention')
            
        Returns:
            Updated conversation details
        """
        response = await self._request("PATCH", f"/admin/conversations/{conversation_id}/{state}")
        return response.get("data", {})
    
    async def rename_conversation(self, conversation_id: str, new_name: str) -> Dict:
        """Rename a conversation.
        
        Args:
            conversation_id: ID of the conversation
            new_name: New name for the conversation
            
        Returns:
            Updated conversation details with new name
        """
        response = await self._request("PATCH", f"/user/conversations/{conversation_id}/rename/{new_name}")
        return response.get("data", {})
        
    # Forum API methods
    
    async def list_forums(self) -> List[Dict]:
        """List all forums accessible to the user.
        
        Returns:
            List of forums
        """
        response = await self._request("GET", "/admin/forum/")
        return response.get("data", {}).get("items", [])
    
    async def list_categories(self, forum_id: str) -> List[Dict]:
        """List all categories in a forum.
        
        Args:
            forum_id: ID of the forum
            
        Returns:
            List of categories
        """
        response = await self._request("GET", f"/admin/forum/{forum_id}/category/")
        return response.get("data", {}).get("items", [])
    
    async def create_thread(
        self, 
        forum_id: str, 
        category_id: str, 
        title: str, 
        summary: str
    ) -> Dict:
        """Create a new thread in a category.
        
        Args:
            forum_id: ID of the forum
            category_id: ID of the category
            title: Thread title
            summary: Thread summary
            
        Returns:
            Created thread details
        """
        data = {
            "title": title,
            "summary": summary
        }
        
        response = await self._request("POST", f"/admin/forum/{forum_id}/thread/{category_id}", data)
        return response.get("data", {})
    
    async def update_thread(
        self, 
        forum_id: str, 
        thread_id: str, 
        title: Optional[str] = None, 
        summary: Optional[str] = None,
        category_id: Optional[str] = None
    ) -> Dict:
        """Update an existing thread.
        
        Args:
            forum_id: ID of the forum
            thread_id: ID of the thread
            title: New thread title (optional)
            summary: New thread summary (optional)
            category_id: New category ID (optional)
            
        Returns:
            Updated thread details
        """
        data = {}
        if title is not None:
            data["title"] = title
        if summary is not None:
            data["summary"] = summary
        if category_id is not None:
            data["category_id"] = category_id
            
        response = await self._request("PATCH", f"/admin/forum/{forum_id}/thread/{thread_id}", data)
        return response.get("data", {})
    
    async def create_thread_conversation(
        self, 
        forum_id: str, 
        thread_id: str, 
        conversation_id: str, 
        tags: List[str]
    ) -> Dict:
        """Link a conversation to a thread with tags.
        
        Args:
            forum_id: ID of the forum
            thread_id: ID of the thread
            conversation_id: ID of the conversation
            tags: List of tag IDs
            
        Returns:
            Created thread conversation details with thread_conversation_id
        """
        # The request data structure matches the ThreadConversationCreate model
        data = {
            "conversation_id": conversation_id,
            "tags": tags
        }
        
        # Using the admin endpoint as shown in your provided API
        response = await self._request("POST", f"/admin/forum/{forum_id}/thread/{thread_id}/conversation", data)
        
        # The response includes a "data" field with the thread conversation details
        result = response.get("data", {})
        
        # Log the response for debugging
        logger.debug(f"Thread conversation created with ID: {result.get('thread_conversation_id')}")
        
        return result
    
    async def add_message_selections(
        self, 
        forum_id: str, 
        thread_id: str, 
        thread_conversation_id: str, 
        message_ids: List[str]
    ) -> Dict:
        """Select messages to include in a thread conversation.
        
        Args:
            forum_id: ID of the forum
            thread_id: ID of the thread
            thread_conversation_id: ID of the thread conversation
            message_ids: List of message IDs to select
            
        Returns:
            Result of the bulk selection operation
        """
        # The request data structure matches the BulkMessageSelectionCreate model
        data = {
            "message_ids": message_ids
        }
        
        # Using the admin endpoint as shown in your provided API
        response = await self._request(
            "POST", 
            f"/admin/forum/{forum_id}/thread/{thread_id}/conversation/{thread_conversation_id}/selection/bulk", 
            data
        )
        
        # The response includes a "data" field with the operation result
        result = response.get("data", {})
        
        # Log the response for debugging
        logger.debug(f"Added {result.get('count', 0)} message selections to thread conversation")
        
        return result
    
    async def get_forum_tags(self) -> List[Dict]:
        """Get all available forum tags.
        
        Returns:
            List of tags
        """
        response = await self._request("GET", "/admin/forum/tag/")
        return response.get("data", {}).get("items", [])
    
    async def create_forum_tag(self, name: str) -> Dict:
        """Create a new forum tag.
        
        Args:
            name: Tag name
            
        Returns:
            Created tag details
        """
        data = {
            "name": name
        }
        
        response = await self._request("POST", "/admin/forum/tag/", data)
        return response.get("data", {})
    
    async def get_or_create_tags(self, tag_names: List[str]) -> List[str]:
        """Get or create forum tags by name.
        
        Args:
            tag_names: List of tag names
            
        Returns:
            List of tag IDs
        """
        if not tag_names:
            return []
            
        # Get existing tags
        try:
            existing_tags = await self.get_forum_tags()
            existing_tag_map = {tag['name'].lower(): tag['tag_id'] for tag in existing_tags}
            
            # Check which tags need to be created
            tag_ids = []
            for name in tag_names:
                if not name:  # Skip empty tag names
                    continue
                    
                # Convert to lowercase for case-insensitive matching
                name_lower = name.lower()
                if name_lower in existing_tag_map:
                    tag_ids.append(existing_tag_map[name_lower])
                    logger.debug(f"Using existing tag: {name} -> {existing_tag_map[name_lower]}")
                else:
                    # Create new tag
                    try:
                        new_tag = await self.create_forum_tag(name)
                        tag_id = new_tag.get('tag_id')
                        if tag_id:
                            tag_ids.append(tag_id)
                            logger.debug(f"Created new tag: {name} -> {tag_id}")
                        else:
                            logger.error(f"Failed to create tag '{name}'. No tag_id returned: {new_tag}")
                    except Exception as e:
                        logger.error(f"Error creating tag '{name}': {str(e)}")
            
            logger.debug(f"Final tag IDs: {tag_ids}")
            return tag_ids
        except Exception as e:
            logger.error(f"Error in get_or_create_tags: {str(e)}")
            return []
    
    async def list_threads(self, forum_id: str, category_id: Optional[str] = None) -> List[Dict]:
        """List all threads in a forum, optionally filtered by category.
        
        Args:
            forum_id: ID of the forum
            category_id: Optional ID of the category to filter by
            
        Returns:
            List of threads
        """
        endpoint = f"/admin/forum/{forum_id}/thread/"
        if category_id:
            endpoint += f"?category_id={category_id}"
            
        response = await self._request("GET", endpoint)
        return response.get("data", {}).get("items", [])

    async def semantic_search_threads(
        self, 
        forum_id: str, 
        query: str, 
        page: int = 1, 
        page_size: int = 10
    ) -> Dict:
        """Perform semantic search on threads within a forum.
        
        Args:
            forum_id: UUID of the forum to search in
            query: Search query text - can be a keyword, sentence, or paragraph
            page: Page number for pagination (default: 1)
            page_size: Number of results per page (default: 10, max: 20)
            
        Returns:
            Search results including thread items and pagination info
        """
        endpoint = f"/admin/forum/{forum_id}/semantic_search?query={query}&page={page}&pageSize={page_size}"
        response = await self._request("GET", endpoint)
        return response.get("data", {}) 
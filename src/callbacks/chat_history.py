import json
import os
import logging
from datetime import datetime
from uuid import uuid4
from typing import Any, Dict, Optional, Tuple

from arshai.clients.arshai.chat_history_client import ChatHistoryClient
from arshai.utils.logging import get_logger

logger = get_logger(__name__)

class ChatHistoryCallbackHandler:
    """Custom callback handler for chat that integrates with the chat history service."""
    
    def __init__(self, 
                message_time=None,
                conversation_id=None, 
                correlation_id=None, 
                request_id=None,
                parent_message=None, 
                user_data=None,
                realm=None,
                agent_title=None,
                is_anonymous=True) -> None:
        """Initialize the chat callback handler.
        
        Args:
            settings: Application settings
            socket_queue: Queue for socket messages
            sse_queue: Queue for SSE messages
            query: User query
            message_time: Time of message
            conversation_id: ID of the conversation
            correlation_id: Correlation ID for tracking
            request_id: Request ID
            parent_message: ID of parent message
            user_data: User data
            realm: Realm
            agent_title: Agent title
        """
        super().__init__()
        self.model_name = None
        self._correlation_id = correlation_id
        self._request_id = request_id
        self._conversation_id = conversation_id
        self._parent_message = parent_message
        self._user_data = user_data
        self._stop_signal = None
        self._realm = realm
        self._agent_title = agent_title
        self._message_time = message_time
        self._is_anonymous = is_anonymous
        # Initialize chat history client
        self.chat_history_client = ChatHistoryClient(header_data=self._user_data)
        
        logger.info("ChatHistoryCallbackHandler initialized")

    async def send_message(self, conversation_id,
                           message_text,
                           sender, 
                           parent_message_id,
                           message_time, 
                           metadata=[],
                           conversation_state = "normal") -> str:
        """Send a message to an existing conversation.
        
        Args:
            conversation_id: ID of the conversation
            message_text: Text content of the message
            is_user_message: Whether this is a user message (vs AI)
            parent_message_id: ID of the parent message
            response_time: Time of response (default: current time)
            metadata: Message metadata
            
        Returns:
            Message ID of the created message
        """

        message_id = str(uuid4())
        
        first_name = self._user_data.get('details', {}).get('given_name')
        last_name = self._user_data.get('details', {}).get('family_name')
        
        message = {
            'version': 0,
            'correlation_id': self._correlation_id,
            'request_id': self._request_id,
            'parent_message_id': parent_message_id,
            'message_id': message_id,
            'model': self.model_name,
            'metadata': metadata,
            'need_human_responder': True if conversation_state == "human-intervention" else False,
            'text': message_text,
            'error': False,
            'is_edited': False,
            'is_viewed': True,
            'is_visible': True,
            'finish_reason': '',
            "created_at": message_time,
            "updated_at": message_time,
        }
        
        if sender == 'end_user':
            message.update({
                'responder': sender,
                'first_name': first_name,
                'last_name': last_name,
                'user_metadata': self._user_data.get('user_metadata', {})
            })
        else:
            message.update({
                'responder': sender
            })
            
        logger.debug(f"Message as {sender}: {message}")
        try:
            data = {
                "conversation_id": conversation_id,
                "state": conversation_state,
                "messages": [message]
            }
            await self.chat_history_client.add_message(data)
            return message_id
        except Exception as e:
            logger.error(f"Error sending message to history service: {str(e)}")
            return message_id

    async def create_conversation(self,
                                  conversation_state = "normal") -> Tuple[str, str]:
        """Create a new conversation with initial user message and bot response.
        
        Args:
            user_message_text: Text of the user's message
            bot_response_text: Text of the bot's response
            response_time: Time of response (default: current time)
            conversation_state: State of the conversation
            metadata: Message metadata
            
        Returns:
            Tuple of (conversation_id, bot_message_id)
        """

        
        user_id = self._user_data.get('user_id')
        first_name = self._user_data.get('details', {}).get('given_name')
        last_name = self._user_data.get('details', {}).get('family_name')
        
        try:
            conversation_id = await self.chat_history_client.create_conversation(
                correlation_id=self._correlation_id,
                org=self._user_data.get('org_id', "27820ae0-d693-42cc-acf0-14064f8a393a"),
                first_name=first_name,
                last_name=last_name,
                conversation_id=self._conversation_id,
                state=conversation_state,
                temperature=0.0,
                model=self.model_name,
                agent_title=self._agent_title,
                messages=[],
                realm=self._realm,
                is_anonymous=self._is_anonymous
            )
            
            logger.info(f"Conversation stored in history service: {conversation_id}")
            return conversation_id
                
        except Exception as e:
            logger.error(f"Error creating conversation in history service: {str(e)}")
            return None

    async def rename_conversation(self, conversation_id: str, new_name: str) -> Dict:
        """Rename an existing conversation.
        
        Args:
            conversation_id: ID of the conversation to rename
            new_name: New name/title for the conversation
            
        Returns:
            Response data from the rename operation
        """
        try:
            result = await self.chat_history_client.rename_conversation(conversation_id, new_name)
            logger.info(f"Conversation {conversation_id} renamed to '{new_name}'")
            return result
        except Exception as e:
            logger.error(f"Error renaming conversation in history service: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_conversation_details(self, conversation_id: str) -> Dict:
        """Get details of an existing conversation including its title/name.
        
        Args:
            conversation_id: ID of the conversation to retrieve
            
        Returns:
            Conversation details including title, state, and other metadata
        """
        try:
            result = await self.chat_history_client.get_conversation(conversation_id)
            logger.info(f"Retrieved details for conversation {conversation_id}")
            return result 
        except Exception as e:
            logger.error(f"Error getting conversation details from history service: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_conversation_state(self, conversation_id: str) -> str:
        """Get the state of a conversation.
        
        Args:
            conversation_id: ID of the conversation
            auth_user_sub: User ID of the authenticated user
        Returns:
            Conversation state (normal, finish, archive, human-intervention, etc.)
        """
        try:
            result = await self.chat_history_client.get_conversation_state(conversation_id)
            logger.info(f"Retrieved state for conversation {conversation_id}")
            return result
        except Exception as e:
            logger.error(f"Error getting conversation state from history service: {str(e)}")
            return None
        
    async def get_latest_message(self, conversation_id: str, num_messages: int = 1) -> Dict:
        """Get the latest message for a conversation.
        
        Args:
            conversation_id: ID of the conversation
            num_messages: Number of latest messages to retrieve (default: 1)
        Returns:
            Latest message or empty dict if no messages
        """
        try:
            result = await self.chat_history_client.get_messages(conversation_id)
            logger.info(f"Retrieved latest message for conversation {conversation_id}")
            return result.get("items", [])[0:num_messages]
        except Exception as e:
            logger.error(f"Error getting latest message from history service: {str(e)}")
            return None

    async def check_conversation_id(self, conversation_id: str) -> bool:
        """Check if a conversation ID is valid and accessible to the current user.
        
        Args:
            conversation_id: ID of the conversation to check
            
        Returns:
            Boolean indicating whether the conversation exists and is accessible
        """
        try:
            result = await self.chat_history_client.check_conversation_id(conversation_id)
            logger.info(f"Checked validity of conversation ID {conversation_id}: {'valid' if result else 'invalid'}")
            return result
        except Exception as e:
            logger.error(f"Error checking conversation ID validity: {str(e)}")
            return False
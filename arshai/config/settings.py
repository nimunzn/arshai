"""
Settings implementation for Arshai using ConfigManager.
"""

import logging
import os
from typing import Any, Dict, Optional, List, Type
from arshai.core.interfaces.isetting import ISetting
from arshai.core.interfaces.illm import ILLM, ILLMConfig
from arshai.core.interfaces.imemorymanager import IMemoryManager
from arshai.core.interfaces.ireranker import IReranker
from arshai.core.interfaces.iagent import IAgent, IAgentConfig
from arshai.core.interfaces.iembedding import IEmbedding
from arshai.core.interfaces.ivector_db_client import IVectorDBClient
from arshai.core.interfaces.iwebsearch import IWebSearchClient
from arshai.core.interfaces.ispeech import ISpeechProcessor, ISpeechConfig

from ..factories.llm_factory import LLMFactory
from ..factories.memory_factory import MemoryFactory
from ..factories.vector_db_factory import VectorDBFactory
from ..factories.reranker_factory import RerankerFactory
from ..factories.agent_factory import AgentFactory
from ..factories.embedding_factory import EmbeddingFactory
from ..factories.search_factory import WebSearchFactory
from ..factories.speech_factory import SpeechFactory

from .config_manager import ConfigManager

logger = logging.getLogger(__name__)


class Settings(ISetting):
    """
    Settings implementation for Arshai.
    
    This class implements the ISetting interface using ConfigManager
    for configuration management. It is designed to be extended by applications
    that build on the Arshai framework.
    
    The Settings class focuses only on structural configuration (which providers to use)
    while sensitive data (API keys, endpoints, URLs) is read directly from environment
    variables by the component implementations themselves.
    
    To extend this class:
    1. Inherit from it in your application
    2. Override any methods you need to customize
    3. Add application-specific methods
    4. Register custom components in __init__ or a dedicated method
    
    Example:
        class BusinessSettings(Settings):
            def __init__(self, config_path=None):
                super().__init__(config_path)
                # Register custom components
                self._register_custom_components()
                
            def _register_custom_components(self):
                # Register custom agents
                from my_app.agents import CustomAgent
                AgentFactory.register("custom_agent", CustomAgent)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize settings with configuration.
        
        Args:
            config_path: Optional path to a configuration file
        """
        self.config_manager = ConfigManager(config_path)
        
        # Debug: Print all configuration fields
        config_fields = self.config_manager.get_all()
        logger.info("Loaded configuration fields:")
        for key, value in config_fields.items():
            logger.info(f"  {key}: {value}")
        
        # Cache for created components
        self._llm = None
        self._memory_manager = None
        self._retriever = None
        self._reranker = None
        self._embedding = None
        self._web_search = None
        self._vector_db = None
        self._speech_processor = None
        self._collection_config = None

    # Property implementations for ISetting interface
    @property
    def chatbot_context(self) -> str:
        """Get the context/instructions for the chatbot"""
        return self.get("chatbot.context", "You are a helpful assistant.")
    
    @property
    def redis_url(self) -> str:
        """Get Redis URL from environment or default"""
        return os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    
    def get_value(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key (dot-separated path)
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        return self.config_manager.get(key, default)
    
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Alias for get_value, for backward compatibility and convenience.
        
        Args:
            key: Configuration key (dot-separated path)
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        return self.get_value(key, default)
    
    def model_dump(self) -> Dict[str, Any]:
        """
        Convert the settings object to a serializable dictionary.
        
        This method is used for serialization when settings need to be passed
        between components or services that require JSON serialization.
        
        Returns:
            Dict containing the serialized settings data
        """
        # Get all configuration data from the config manager
        config_data = self.config_manager.get_all()
        
        # Create a clean dictionary with only serializable data
        serializable_data = {}
        
        # Include essential configuration that might be needed by workers
        serializable_data["config"] = config_data
        
        # Include any other essential settings properties needed by workers
        serializable_data["llm"] = {
            "provider": self.get("llm.provider"),
            "model": self.get("llm.model"),
            "temperature": self.get("llm.temperature"),
            "max_tokens": self.get("llm.max_tokens")
        }
        
        # Include memory configuration
        serializable_data["memory"] = self.get("memory", {})
        
        # Include retriever configuration
        serializable_data["retriever"] = self.get("retriever", {})
        
        # Include embedding configuration
        serializable_data["embedding"] = self.get("embedding", {})
        
        # Include vector_db configuration if available
        serializable_data["vector_db"] = self.get("vector_db", {})
        
        # Include speech configuration if available
        serializable_data["speech"] = self.get("speech", {})
        
        return serializable_data
    
    def create_llm(self) -> ILLM:
        """
        Create an LLM instance based on configuration.
        
        This method passes only structural configuration to the LLM factory.
        The LLM implementation is responsible for reading sensitive data
        (API keys, endpoints) directly from environment variables.
        
        Override this method in a subclass to customize LLM creation.
        
        Returns:
            ILLM: An instance of a language model
        """
        if self._llm:
            return self._llm
            
        provider = self.get("llm.provider", "azure")
        model = self.get("llm.model", "gpt-4-mini")
        temperature = self.get("llm.temperature", 0.0)
        
        # Only include max_tokens if specified in config
        max_tokens = self.get("llm.max_tokens")
        
        config_params = {
            "model": model,
            "temperature": temperature
        }
        
        # Only add max_tokens if provided
        if max_tokens is not None:
            config_params["max_tokens"] = max_tokens
            
        config = ILLMConfig(**config_params)
        
        # Create LLM with only structural configuration
        # No sensitive data is passed here
        self._llm = LLMFactory.create(provider, config)
        return self._llm
    
    def create_memory_manager(self) -> IMemoryManager:
        """
        Create a memory manager based on configuration.
        
        This method passes only structural configuration to the memory factory.
        The memory implementation is responsible for reading sensitive data
        (connection URLs, credentials) directly from environment variables.
        
        Override this method in a subclass to customize memory manager creation.
        
        Returns:
            IMemoryManager: An instance of a memory manager
        """
        if self._memory_manager:
            return self._memory_manager
            
        # Only pass structural configuration
        memory_config = self.get("memory", {})
        self._memory_manager = MemoryFactory.create_memory_manager_service(memory_config)
        return self._memory_manager
    

    def create_vector_db(self) -> Optional[IVectorDBClient]:
        """
        Create a vector database client based on configuration.
        
        This method passes only structural configuration to the vector database factory.
        The vector database implementation is responsible for reading sensitive data
        directly from environment variables.
        
        Importantly, this method will also initialize the retriever for the vector database
        client using the embedding model and reranker from settings. This means that if
        you call both `create_vector_db()` and `create_retriever()`, you'll get two different
        retriever instances. In most cases, you'll want to use the retriever from the
        vector database client, which you can access via `vector_db.retriever`.
        
        Override this method in a subclass to customize vector database creation.
        
        Returns:
            Optional[IVectorDBClient]: An instance of a vector database client, or None if not configured
        """
        if self._vector_db:
            return self._vector_db, self._collection_config, self._embedding
    
        self._embedding = self.create_embedding()
        logger.info(f"embedding model created {self._embedding}")
        # Create vector database with appropriate configuration objects based on provider
        
        self._vector_db, self._collection_config = VectorDBFactory.create(embedding_dimension=self._embedding.dimension, settings=self)
        return self._vector_db, self._collection_config, self._embedding
    
    def create_web_search(self) -> Optional[IWebSearchClient]:
        """
        Create a web search component based on configuration.
        
        This method uses the dedicated SearchFactory to create search components
        like SearxNG. Search implementations can have different providers but 
        all serve the purpose of retrieving information from the web.
        
        Returns:
            Optional[IWebSearchClient]: A web search component instance, or None if not configured
        """
        if self._web_search:
            return self._web_search
            
        # First check for dedicated search config
        search_config = self.get("search")
        
        # Fall back to web_search if needed
        if not search_config:
            search_config = self.get("web_search")
            
        if not search_config:
            logger.warning("No search configuration found")
            return None
              
        # Get the provider from config, default to searxng
        provider = search_config.get("provider", "searxng")
            
        try:
            # Pass self (the settings instance) to the factory
            self._web_search = WebSearchFactory.create(provider, search_config, settings=self)
            logger.info(f"Created web search component with provider: {provider}")
            return self._web_search
        except Exception as e:
            logger.error(f"Error creating web search component: {str(e)}")
            return None
    
    def create_reranker(self) -> Optional[IReranker]:
        """
        Create a reranker based on configuration if configured.
        
        This method passes only structural configuration to the reranker factory.
        The reranker implementation is responsible for reading sensitive data
        directly from environment variables.
        
        Override this method in a subclass to customize reranker creation.
        
        Returns:
            Optional[IReranker]: An instance of a reranker, or None if not configured
        """
        if self._reranker:
            return self._reranker
            
        reranker_config = self.get("reranker")
        if not reranker_config:
            logger.warning("No reranker configuration found")
            return None
            
        provider = reranker_config.get("provider")
        if not provider:
            logger.warning("No reranker provider specified in configuration")
            return None
            
        # Create reranker with only structural configuration
        try:
            self._reranker = RerankerFactory.create(provider, reranker_config)
            logger.info(f"Created reranker with provider: {provider}")
            return self._reranker
        except Exception as e:
            logger.error(f"Error creating reranker: {str(e)}")
            return None
    
    def create_agent(self, agent_type: str, config: IAgentConfig) -> IAgent:
        """
        Create an agent of the specified type with the given configuration.
        
        This method uses the AgentFactory to create predefined agents. 
        For custom agents, it's recommended to directly instantiate them 
        instead of using this factory method.
        
        Args:
            agent_type: Type of agent to create (e.g., "conversation")
            config: Configuration for the agent
            
        Returns:
            IAgent: An instance of the specified agent type
            
        Raises:
            ValueError: If the agent type is not supported
            
        Example:
            config = IAgentConfig(
                task_context="You are a helpful assistant",
                tools=[]
            )
            agent = settings.create_agent("conversation", config)
        """
        return AgentFactory.create(agent_type, config, self)
    
    def create_embedding(self) -> Optional[IEmbedding]:
        """
        Create an embedding service based on configuration.
        
        This method passes only structural configuration to the embedding factory.
        The embedding implementation is responsible for reading sensitive data
        directly from environment variables.
        
        Override this method in a subclass to customize embedding creation.
        
        Returns:
            Optional[IEmbedding]: An instance of an embedding service, or None if not configured
        """
        if self._embedding:
            return self._embedding
            
        embedding_config = self.get("embedding")
        if not embedding_config:
            logger.warning("No embedding configuration found")
            return None
            
        provider = embedding_config.get("provider", "mgte")
        # Create embedding with only structural configuration
        try:
            logger.info(f"Creating embedding model with provider: {provider}, config: {embedding_config}")
            self._embedding = EmbeddingFactory.create(provider, embedding_config)
            logger.info(f"Created embedding service with provider: {provider}")
            return self._embedding
        except Exception as e:
            logger.error(f"Error creating embedding service: {str(e)}")
            return None
            
    def create_speech_model(self) -> Optional[ISpeechProcessor]:
        """
        Create a speech model based on configuration.
        
        This method passes only structural configuration to the speech factory.
        The speech model implementation is responsible for reading sensitive data
        (API keys, endpoints) directly from environment variables.
        
        Override this method in a subclass to customize speech model creation.
        
        Returns:
            Optional[ISpeechProcessor]: An instance of a speech model, or None if not configured
        """
        if self._speech_processor:
            return self._speech_processor
        
        speech_config = self.get("speech")
        if not speech_config:
            logger.warning("No speech configuration found")
            return None
        
        provider = speech_config.get("provider", "openai")
        
        # Create speech config object with only structural parameters
        config_params = {
            "stt_model": speech_config.get("stt_model", "whisper-1"),
            "tts_model": speech_config.get("tts_model"),
            "tts_voice": speech_config.get("tts_voice"),
            "region": speech_config.get("region")
        }
        
        speech_config_obj = ISpeechConfig(**config_params)
        
        # Create speech processor with only structural configuration
        try:
            self._speech_processor = SpeechFactory.create(provider, speech_config_obj)
            logger.info(f"Created speech processor with provider: {provider}")
            return self._speech_processor
        except Exception as e:
            logger.error(f"Error creating speech processor: {str(e)}")
            return None 
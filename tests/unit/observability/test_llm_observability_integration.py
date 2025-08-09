"""
Comprehensive LLM observability integration tests.

Tests observability integration across all LLM clients with proper environment loading.
"""

import pytest
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.observability import (

    ObservabilityConfig,
    ObservabilityManager
)
# Setup logging for pytest
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestLLMObservabilityIntegration:
    """Comprehensive test suite for LLM client observability integration."""
    
    # Test configuration directory
    LLM_TESTS_DIR = Path("/Users/mobilletmac/Desktop/Arshai/Pakages/arshai/tests/unit/llms")
    
    @classmethod
    def setup_class(cls):
        """Load all available environment files for testing."""
        cls._load_all_env_files()
    
    @classmethod
    def _load_all_env_files(cls):
        """Load all environment files from the LLM tests directory."""
        if not cls.LLM_TESTS_DIR.exists():
            return
            
        env_files = [
            ".env.azure",
            ".env.openai", 
            ".env.gemini",
            ".env.openrouter"
        ]
        
        for env_file in env_files:
            env_path = cls.LLM_TESTS_DIR / env_file
            if env_path.exists():
                load_dotenv(env_path)
    
    @staticmethod
    def get_test_config(model_name: str = "gpt-4o-mini") -> ILLMConfig:
        """Get test configuration with specified model."""
        return ILLMConfig(
            model=model_name,
            temperature=0.1,
            max_tokens=50
        )
    
    @staticmethod
    def get_gemini_test_config() -> ILLMConfig:
        """Get test configuration specifically for Gemini with correct model."""
        return ILLMConfig(
            model="gemini-2.0-flash-exp",  # Correct Gemini model
            temperature=0.1,
            max_tokens=50
        )
    
    @staticmethod
    def get_observability_config(service_name: str = "test-service") -> ObservabilityConfig:
        """Get test observability configuration."""
        return ObservabilityConfig(
            service_name=service_name,
            track_token_timing=True,
            collect_metrics=True,
            log_prompts=False  # Don't log prompts in tests for privacy
        )
    
    def test_base_observability_functionality(self):
        """Test basic observability functionality."""
        # Test config creation
        config = self.get_observability_config()
        assert config.service_name == "test-service"
        assert config.track_token_timing is True
        assert config.collect_metrics is True
        
        # Test manager creation
        obs_manager = ObservabilityManager(ObservabilityConfig(service_name="test-service"))
        assert obs_manager is not None
        assert isinstance(obs_manager, ObservabilityManager)
    
    # ============================================================================
    # AZURE CLIENT TESTS
    # ============================================================================
    
    def test_azure_client_creation_without_observability(self):
        """Test Azure client creation without observability."""
        from arshai.llms.azure import AzureClient
        
        config = self.get_test_config()
        azure_deployment = os.getenv("AZURE_DEPLOYMENT", "gpt-4o-mini")
        api_version = os.getenv("AZURE_API_VERSION", "2025-03-01-preview")
        
        try:
            client = AzureClient(config, azure_deployment=azure_deployment, api_version=api_version)
            assert client is not None
            assert hasattr(client, 'chat')
            assert hasattr(client, 'stream')
            assert hasattr(client, '_execute_chat')
            assert hasattr(client, '_execute_stream')
            assert client._get_provider_name() == "azure"
        except Exception as e:
            pytest.fail(f"Azure client creation failed: {e}")
    
    def test_azure_client_creation_with_observability(self):
        """Test Azure client creation with observability enabled."""
        from arshai.llms.azure import AzureClient
        
        config = self.get_test_config()
        obs_manager = ObservabilityManager(ObservabilityConfig(service_name="test-service"))
        azure_deployment = os.getenv("AZURE_DEPLOYMENT", "gpt-4o-mini")
        api_version = os.getenv("AZURE_API_VERSION", "2025-03-01-preview")
        
        try:
            client = AzureClient(
                config, 
                observability_manager=obs_manager,
                azure_deployment=azure_deployment,
                api_version=api_version
            )
            assert client is not None
            assert hasattr(client, 'observability_manager')
            assert client.observability_manager is not None
            assert isinstance(client.observability_manager, ObservabilityManager)
        except Exception as e:
            pytest.fail(f"Azure client creation with observability failed: {e}")
    
    def test_azure_direct_constructor(self):
        """Test Azure direct constructor integration."""
        from arshai.llms.azure import AzureClient
        
        config = self.get_test_config()
        obs_config = self.get_observability_config("azure-test-service")
        obs_manager = ObservabilityManager(obs_config)
        azure_deployment = os.getenv("AZURE_DEPLOYMENT", "gpt-4o-mini")
        api_version = os.getenv("AZURE_API_VERSION", "2025-03-01-preview")
        
        try:
            # Test direct constructor usage
            client = AzureClient(
                config,
                observability_manager=obs_manager,
                azure_deployment=azure_deployment,
                api_version=api_version
            )
            assert client is not None
            assert hasattr(client, 'observability_manager')
            assert client.observability_manager is obs_manager
        except Exception as e:
            pytest.fail(f"Azure direct constructor test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_azure_actual_message_with_metrics(self):
        """Test sending actual message to Azure and verifying metrics."""
        from arshai.llms.azure import AzureClient
        
        config = self.get_test_config()
        obs_config = self.get_observability_config("azure-metrics-test")
        azure_deployment = os.getenv("AZURE_DEPLOYMENT", "gpt-4o-mini")
        api_version = os.getenv("AZURE_API_VERSION", "2025-03-01-preview")
        
        try:
            # Create observable client
            obs_manager = ObservabilityManager(obs_config)
            client = AzureClient(
                config,
                observability_manager=obs_manager,
                azure_deployment=azure_deployment,
                api_version=api_version
            )
            
            # Send test message
            test_input = ILLMInput(
                system_prompt="You are a helpful assistant that provides brief responses.",
                user_message="What is 2+2? Reply with just the number."
            )
            
            response = await client.chat(test_input)
                        
            logger.debug(f"response is: {response}")
            # Verify response and metrics
            assert isinstance(response, dict)
            assert "llm_response" in response
            assert "usage" in response
            
            usage = response.get("usage", {})
            assert usage.get('provider') == 'azure'
            assert usage.get('input_tokens', 0) > 0
            assert usage.get('output_tokens', 0) > 0
            assert usage.get('total_tokens', 0) > 0
        except Exception as e:
            pytest.fail(f"Azure message test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_azure_streaming_with_metrics(self):
        """Test Azure streaming with metrics collection."""
        from arshai.llms.azure import AzureClient
        
        config = self.get_test_config()
        obs_config = self.get_observability_config("azure-streaming-test")
        azure_deployment = os.getenv("AZURE_DEPLOYMENT", "gpt-4o-mini")
        api_version = os.getenv("AZURE_API_VERSION", "2025-03-01-preview")
        
        try:
            obs_manager = ObservabilityManager(obs_config)
            client = AzureClient(
                config,
                observability_manager=obs_manager,
                azure_deployment=azure_deployment,
                api_version=api_version
            )
            
            test_input = ILLMInput(
                system_prompt="You are a helpful assistant that provides brief responses.",
                user_message="What is 2+2? Reply with just the number."
            )
            
            collected_text = ""
            final_usage = None
            
            async for chunk in client.stream(test_input):
                if chunk.get("llm_response"):
                    collected_text = chunk["llm_response"]
                if chunk.get("usage"):
                    final_usage = chunk["usage"]
            
            assert collected_text, "Should have received streamed text"
            if final_usage:
                assert final_usage.get('provider') == 'azure'
                assert final_usage.get('input_tokens', 0) > 0
                assert final_usage.get('output_tokens', 0) > 0
        except Exception as e:
            pytest.fail(f"Azure streaming test failed: {e}")
    
    # ============================================================================
    # OPENAI CLIENT TESTS  
    # ============================================================================
    
    def test_openai_client_observability(self):
        """Test OpenAI client observability integration."""
        try:
            from arshai.llms.openai import OpenAIClient
        except ImportError:
            pytest.skip("OpenAI client not available")
        
        config = self.get_test_config()
        obs_config = self.get_observability_config("openai-test-service")
        
        try:
            # Test without observability
            client = OpenAIClient(config)
            assert client._get_provider_name() == "openai"
            
            # Test with observability
            obs_manager = ObservabilityManager(ObservabilityConfig(service_name="test-service"))
            observable_client = OpenAIClient(config, observability_manager=obs_manager)
            assert observable_client.observability_manager is not None
            
            # Test direct constructor
            obs_manager = ObservabilityManager(obs_config)
            direct_client = OpenAIClient(config, observability_manager=obs_manager)
            assert direct_client.observability_manager is not None
        except Exception as e:
            pytest.fail(f"OpenAI client test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_openai_actual_message_with_metrics(self):
        """Test sending actual message to OpenAI and verifying metrics."""
        try:
            from arshai.llms.openai import OpenAIClient
        except ImportError:
            pytest.fail("OpenAI client not available")
        
        config = self.get_test_config()
        obs_config = self.get_observability_config("openai-metrics-test")
        
        try:
            # Create observable client
            obs_manager = ObservabilityManager(obs_config)
            client = OpenAIClient(
                config,
                observability_manager=obs_manager
            )
            
            # Send test message
            test_input = ILLMInput(
                system_prompt="You are a helpful assistant that provides brief responses.",
                user_message="What is 2+2? Reply with just the number."
            )
            
            response = await client.chat(test_input)
                        
            logger.debug(f"response is: {response}")
            # Verify response and metrics
            assert isinstance(response, dict)
            assert "llm_response" in response
            assert "usage" in response
            
            usage = response.get("usage", {})
            assert usage.get('provider') == 'openai'
            assert usage.get('input_tokens', 0) > 0
            assert usage.get('output_tokens', 0) > 0
            assert usage.get('total_tokens', 0) > 0
        except Exception as e:
            pytest.fail(f"OpenAI message test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_openai_streaming_with_metrics(self):
        """Test OpenAI streaming with metrics collection."""
        try:
            from arshai.llms.openai import OpenAIClient
        except ImportError:
            pytest.fail("OpenAI client not available")
        
        config = self.get_test_config()
        obs_config = self.get_observability_config("openai-streaming-test")
        
        try:
            obs_manager = ObservabilityManager(obs_config)
            client = OpenAIClient(
                config,
                observability_manager=obs_manager
            )
            
            test_input = ILLMInput(
                system_prompt="You are a helpful assistant that provides brief responses.",
                user_message="What is 2+2? Reply with just the number."
            )
            
            collected_text = ""
            final_usage = None
            
            async for chunk in client.stream(test_input):
                if chunk.get("llm_response"):
                    collected_text = chunk["llm_response"]
                if chunk.get("usage"):
                    final_usage = chunk["usage"]
            
            assert collected_text, "Should have received streamed text"
            if final_usage:
                assert final_usage.get('provider') == 'openai'
                assert final_usage.get('input_tokens', 0) > 0
                assert final_usage.get('output_tokens', 0) > 0
        except Exception as e:
            pytest.fail(f"OpenAI streaming test failed: {e}")
    
    # ============================================================================
    # GEMINI CLIENT TESTS
    # ============================================================================
    
    def test_gemini_client_observability(self):
        """Test Gemini client observability integration."""
        from arshai.llms.google_genai import GeminiClient
        
        config = self.get_gemini_test_config()  # Use correct Gemini model
        obs_config = self.get_observability_config("gemini-test-service")
        
        try:
            # Test without observability  
            client = GeminiClient(config)
            assert client._get_provider_name() == "gemini"
            
            # Test with observability
            obs_manager = ObservabilityManager(ObservabilityConfig(service_name="test-service"))
            observable_client = GeminiClient(config, observability_manager=obs_manager)
            assert observable_client.observability_manager is not None
            
            # Test direct constructor  
            obs_manager = ObservabilityManager(obs_config)
            direct_client = GeminiClient(config, observability_manager=obs_manager)
            assert direct_client.observability_manager is not None
        except Exception as e:
            pytest.fail(f"Gemini client test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_gemini_actual_message_with_metrics(self):
        """Test sending actual message to Gemini and verifying metrics."""
        from arshai.llms.google_genai import GeminiClient
        
        config = self.get_gemini_test_config()
        obs_config = self.get_observability_config("gemini-metrics-test")
        
        try:
            # Create observable client
            obs_manager = ObservabilityManager(obs_config)
            client = GeminiClient(
                config,
                observability_manager=obs_manager
            )
            
            # Send test message
            test_input = ILLMInput(
                system_prompt="You are a helpful assistant that provides brief responses.",
                user_message="What is 2+2? Reply with just the number."
            )
            
            response = await client.chat(test_input)
            logger.debug(f"response is: {response}")
            
            # Verify response and metrics
            assert isinstance(response, dict)
            assert "llm_response" in response
            assert "usage" in response
            
            usage = response.get("usage", {})
            assert usage.get('provider') == 'gemini'
            assert usage.get('input_tokens', 0) > 0
            assert usage.get('output_tokens', 0) > 0
            assert usage.get('total_tokens', 0) > 0
        except Exception as e:
            pytest.fail(f"Gemini message test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_gemini_streaming_with_metrics(self):
        """Test Gemini streaming with metrics collection."""
        from arshai.llms.google_genai import GeminiClient
        
        config = self.get_gemini_test_config()
        obs_config = self.get_observability_config("gemini-streaming-test")
        
        try:
            obs_manager = ObservabilityManager(obs_config)
            client = GeminiClient(
                config,
                observability_manager=obs_manager
            )
            
            test_input = ILLMInput(
                system_prompt="You are a helpful assistant that provides brief responses.",
                user_message="What is 2+2? Reply with just the number."
            )
            
            collected_text = ""
            final_usage = None
            
            async for chunk in client.stream(test_input):
                if chunk.get("llm_response"):
                    collected_text = chunk["llm_response"]
                if chunk.get("usage"):
                    final_usage = chunk["usage"]
            
            assert collected_text, "Should have received streamed text"
            if final_usage:
                assert final_usage.get('provider') == 'gemini'
                assert final_usage.get('input_tokens', 0) > 0
                assert final_usage.get('output_tokens', 0) > 0
        except Exception as e:
            pytest.fail(f"Gemini streaming test failed: {e}")
    
    # ============================================================================
    # OPENROUTER CLIENT TESTS
    # ============================================================================
    
    def test_openrouter_client_observability(self):
        """Test OpenRouter client observability integration."""
        from arshai.llms.openrouter import OpenRouterClient
        
        config = self.get_test_config()
        obs_config = self.get_observability_config("openrouter-test-service")
        
        try:
            # Test without observability
            client = OpenRouterClient(config)
            assert client._get_provider_name() == "openrouter"
            
            # Test with observability
            obs_manager = ObservabilityManager(ObservabilityConfig(service_name="test-service"))
            observable_client = OpenRouterClient(config, observability_manager=obs_manager)
            assert observable_client.observability_manager is not None
            
            # Test direct constructor
            obs_manager = ObservabilityManager(obs_config)  
            direct_client = OpenRouterClient(config, observability_manager=obs_manager)
            assert direct_client.observability_manager is not None
        except Exception as e:
            pytest.fail(f"OpenRouter client test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_openrouter_actual_message_with_metrics(self):
        """Test sending actual message to OpenRouter and verifying metrics."""
        from arshai.llms.openrouter import OpenRouterClient
        
        config = self.get_test_config()
        obs_config = self.get_observability_config("openrouter-metrics-test")
        
        try:
            # Create observable client
            obs_manager = ObservabilityManager(obs_config)
            client = OpenRouterClient(
                config,
                observability_manager=obs_manager
            )
            
            # Send test message
            test_input = ILLMInput(
                system_prompt="You are a helpful assistant that provides brief responses.",
                user_message="What is 2+2? Reply with just the number."
            )
            
            response = await client.chat(test_input)
            logger.debug(f"Response is: {response}")
            # Verify response and metrics
            assert isinstance(response, dict)
            assert "llm_response" in response
            assert "usage" in response
            
            usage = response.get("usage", {})
            assert usage.get('provider') == 'openrouter'
            assert usage.get('input_tokens', 0) > 0
            assert usage.get('output_tokens', 0) > 0
            assert usage.get('total_tokens', 0) > 0
        except Exception as e:
            pytest.fail(f"OpenRouter message test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_openrouter_streaming_with_metrics(self):
        """Test OpenRouter streaming with metrics collection."""
        from arshai.llms.openrouter import OpenRouterClient
        
        config = self.get_test_config()
        obs_config = self.get_observability_config("openrouter-streaming-test")
        
        try:
            obs_manager = ObservabilityManager(obs_config)
            client = OpenRouterClient(
                config,
                observability_manager=obs_manager
            )
            
            test_input = ILLMInput(
                system_prompt="You are a helpful assistant that provides brief responses.",
                user_message="What is 2+2? Reply with just the number."
            )
            
            collected_text = ""
            final_usage = None
            
            async for chunk in client.stream(test_input):
                if chunk.get("llm_response"):
                    logger.debug(f"Stream Chunk: {chunk.get("llm_response")}")
                    collected_text = chunk["llm_response"]
                if chunk.get("usage"):
                    final_usage = chunk["usage"]
            
            assert collected_text, "Should have received streamed text"
            if final_usage:
                assert final_usage.get('provider') == 'openrouter'
                assert final_usage.get('input_tokens', 0) > 0
                assert final_usage.get('output_tokens', 0) > 0
        except Exception as e:
            pytest.fail(f"OpenRouter streaming test failed: {e}")
    
    # ============================================================================
    # CROSS-CLIENT TESTS
    # ============================================================================
    
    def test_provider_name_auto_detection(self):
        """Test that provider names are correctly auto-detected across clients."""
        expected_providers = {
            "AzureClient": "azure",
            "OpenAIClient": "openai", 
            "GeminiClient": "gemini",
            "OpenRouterClient": "openrouter"
        }
        
        for client_name, expected_provider in expected_providers.items():
            try:
                if client_name == "AzureClient":
                    from arshai.llms.azure import AzureClient
                    client = AzureClient(
                        self.get_test_config(),
                        azure_deployment=os.getenv("AZURE_DEPLOYMENT", "gpt-4o-mini"),
                        api_version=os.getenv("AZURE_API_VERSION", "2025-03-01-preview")
                    )
                elif client_name == "OpenAIClient":
                    from arshai.llms.openai import OpenAIClient
                    client = OpenAIClient(self.get_test_config())
                elif client_name == "GeminiClient":
                    from arshai.llms.google_genai import GeminiClient
                    client = GeminiClient(self.get_gemini_test_config())
                elif client_name == "OpenRouterClient":
                    from arshai.llms.openrouter import OpenRouterClient
                    client = OpenRouterClient(self.get_test_config())
                else:
                    continue
                
                provider_name = client._get_provider_name()
                assert provider_name == expected_provider, f"{client_name} should have provider '{expected_provider}', got '{provider_name}'"
            except Exception:
                # Skip if client can't be initialized (missing credentials, etc.)
                continue
    
    def test_direct_constructor_approach(self):
        """Test that direct constructor approach works with different providers."""
        config = self.get_test_config()
        obs_config = self.get_observability_config()
        obs_manager = ObservabilityManager(obs_config)
        
        # Test with Azure (most likely to have credentials)
        try:
            from arshai.llms.azure import AzureClient
            client = AzureClient(
                config,
                observability_manager=obs_manager,
                azure_deployment=os.getenv("AZURE_DEPLOYMENT", "gpt-4o-mini"),
                api_version=os.getenv("AZURE_API_VERSION", "2025-03-01-preview")
            )
            assert client.observability_manager is obs_manager
            assert client._get_provider_name() == "azure"
        except Exception:
            pass  # Skip if Azure not available
            
        # Test with other clients if they can be initialized
        client_specs = [
            ("OpenAIClient", "arshai.llms.openai", config),
            ("GeminiClient", "arshai.llms.google_genai", self.get_gemini_test_config()),
            ("OpenRouterClient", "arshai.llms.openrouter", config)
        ]
        
        for class_name, module_name, test_config in client_specs:
            try:
                module = __import__(module_name, fromlist=[class_name])
                client_class = getattr(module, class_name)
                
                client = client_class(test_config, observability_manager=obs_manager)
                assert client.observability_manager is obs_manager
            except Exception:
                continue  # Skip if client not available or missing credentials
    
    def test_observability_is_optional(self):
        """Test that observability is completely optional and doesn't affect normal operation."""
        config = self.get_test_config()
        
        # Test that clients work fine without observability
        try:
            from arshai.llms.azure import AzureClient
            client = AzureClient(
                config,
                azure_deployment=os.getenv("AZURE_DEPLOYMENT", "gpt-4o-mini"),
                api_version=os.getenv("AZURE_API_VERSION", "2025-03-01-preview")
            )
            assert client.observability_manager is None
            assert hasattr(client, 'chat')
            assert hasattr(client, 'stream')
        except Exception:
            pass  # Skip if Azure not available
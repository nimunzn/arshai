"""
Safe HTTP client factory with version compatibility and graceful degradation.
Resistant to package upgrades and dependency changes.

This module provides a robust factory for creating HTTP clients that work across
different package versions and gracefully degrade when advanced features are
not available.
"""

import logging
import sys
from typing import Any, Optional, Dict, Union

logger = logging.getLogger(__name__)


class SafeHttpClientFactory:
    """Factory for creating HTTP clients with safe defaults and version compatibility"""
    
    @staticmethod
    def create_openai_client(api_key: str, use_custom_http_client: bool = True, **kwargs) -> Any:
        """
        Create OpenAI client with safe configuration and version compatibility.
        
        Args:
            api_key: OpenAI API key
            **kwargs: Additional arguments to pass to OpenAI client
            
        Returns:
            Configured OpenAI client instance
            
        Raises:
            ValueError: If client cannot be created with any configuration
        """
        
        # Strategy 1: Try optimized configuration
        try:
            client = SafeHttpClientFactory._create_optimized_openai_client(api_key, **kwargs)
            logger.info("Created OpenAI client with optimized configuration")
            return client
        except Exception as e:
            logger.warning(f"Optimized OpenAI client failed: {e}, trying standard config")
        
        # Strategy 2: Try standard configuration
        try:
            client = SafeHttpClientFactory._create_standard_openai_client(api_key, **kwargs)
            logger.info("Created OpenAI client with standard configuration")
            return client
        except Exception as e:
            logger.warning(f"Standard OpenAI client failed: {e}, using minimal config")
        
        # Strategy 3: Minimal fallback
        try:
            client = SafeHttpClientFactory._create_minimal_openai_client(api_key, **kwargs)
            logger.info("Created OpenAI client with minimal configuration")
            return client
        except Exception as e:
            logger.error(f"All OpenAI client configurations failed: {e}")
            raise ValueError(f"Cannot initialize OpenAI client: {e}")
    
    @staticmethod
    def _create_optimized_openai_client(api_key: str, **kwargs) -> Any:
        """High performance configuration with connection pooling"""
        try:
            # Check if httpx is available and get version
            import httpx
            httpx_version = getattr(httpx, '__version__', '0.0.0')
            logger.debug(f"Using httpx version: {httpx_version}")
            
            # Version-compatible httpx configuration
            limits_config = SafeHttpClientFactory._get_safe_limits_config(httpx_version)
            timeout_config = SafeHttpClientFactory._get_safe_timeout_config(httpx_version)
            additional_config = SafeHttpClientFactory._get_additional_httpx_config(httpx_version)
            
            safe_http_client = httpx.Client(
                limits=limits_config,
                timeout=timeout_config,
                verify=True,  # SSL verification enabled for security
                **additional_config
            )
            
            # Import OpenAI with version check
            from openai import OpenAI
            openai_version = SafeHttpClientFactory._get_package_version('openai')
            logger.debug(f"Using OpenAI version: {openai_version}")
            
            # Version-compatible OpenAI client configuration
            openai_config = SafeHttpClientFactory._get_openai_client_config(openai_version)
            
            client_kwargs = {
                'api_key': api_key,
                'http_client': safe_http_client,
                **openai_config,
                **kwargs
            }
            
            return OpenAI(**client_kwargs)
            
        except ImportError as e:
            logger.warning(f"Required packages not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Optimized client creation failed: {e}")
            raise
    
    @staticmethod
    def _get_safe_limits_config(httpx_version: str) -> Any:
        """Get httpx.Limits configuration compatible with version"""
        try:
            import httpx
            
            # Default safe configuration - conservative limits to prevent connection exhaustion
            config = {
                'max_connections': 50,  # Reduced from 200 to prevent exhaustion
                'max_keepalive_connections': 20,  # Reduced from 100
            }
            
            # Version-specific adjustments
            try:
                major_version = int(httpx_version.split('.')[0]) if httpx_version else 0
                
                if major_version >= 1:
                    # httpx 1.x+ configuration
                    config['keepalive_expiry'] = 30.0  # Reduced from 60 to close idle connections sooner
                elif major_version == 0:
                    # httpx 0.x configuration (legacy support)
                    # Some parameters might have different names in older versions
                    pass
            except (ValueError, IndexError):
                logger.warning(f"Could not parse httpx version: {httpx_version}")
            
            return httpx.Limits(**config)
            
        except Exception as e:
            logger.warning(f"Could not create Limits config: {e}")
            # Fallback to basic configuration
            try:
                import httpx
                return httpx.Limits(max_connections=30, max_keepalive_connections=15)
            except Exception:
                # If even basic config fails, let it raise
                raise
    
    @staticmethod
    def _get_safe_timeout_config(httpx_version: str) -> Any:
        """Get httpx.Timeout configuration compatible with version"""
        try:
            import httpx
            
            # Try detailed timeout configuration
            return httpx.Timeout(
                connect=10.0,  # Connection timeout
                read=60.0,     # Read timeout (important for streaming)
                write=10.0,    # Write timeout
                pool=5.0       # Pool acquisition timeout
            )
        except Exception as e:
            logger.warning(f"Could not create detailed Timeout config: {e}")
            # Fallback to simple timeout
            try:
                import httpx
                return httpx.Timeout(30.0)
            except Exception:
                # Final fallback - return simple number
                return 30.0
    
    @staticmethod
    def _get_additional_httpx_config(httpx_version: str) -> Dict[str, Any]:
        """Get additional httpx configuration based on version"""
        config = {}
        
        try:
            major_version = int(httpx_version.split('.')[0]) if httpx_version else 0
            
            # Add version-specific configurations
            if major_version >= 1:
                # httpx 1.x+ features
                try:
                    config['follow_redirects'] = True
                except Exception:
                    pass
                
                # Add trust_env for better compatibility
                try:
                    config['trust_env'] = True
                except Exception:
                    pass
            
        except (ValueError, IndexError):
            logger.warning(f"Could not determine httpx version config for: {httpx_version}")
        except Exception as e:
            logger.warning(f"Error setting additional httpx config: {e}")
        
        return config
    
    @staticmethod
    def _get_openai_client_config(openai_version: str) -> Dict[str, Any]:
        """Get OpenAI client configuration based on version"""
        config = {}
        
        try:
            # Parse version
            version_parts = openai_version.split('.')
            major_version = int(version_parts[0]) if version_parts else 1
            
            # Version-specific configurations
            if major_version >= 1:
                config['max_retries'] = 3
                # Add other v1+ specific configs as they become available
            else:
                # Legacy version support
                config['max_retries'] = 2
            
        except (ValueError, IndexError):
            logger.warning(f"Could not parse OpenAI version: {openai_version}")
            # Safe defaults
            config['max_retries'] = 2
        except Exception as e:
            logger.warning(f"Error determining OpenAI version config: {e}")
            config['max_retries'] = 2
        
        return config
    
    @staticmethod
    def create_azure_openai_client(azure_deployment: str, api_version: str, use_custom_http_client: bool = True, **kwargs) -> Any:
        """
        Create Azure OpenAI client with safe configuration and version compatibility.
        
        Args:
            azure_deployment: Azure deployment name
            api_version: Azure API version
            use_custom_http_client: Whether to try using custom httpx client
            **kwargs: Additional arguments to pass to AzureOpenAI client
            
        Returns:
            Configured AzureOpenAI client instance
            
        Raises:
            ValueError: If client cannot be created with any configuration
        """
        
        # Strategy 1: Try with custom httpx client if requested
        if use_custom_http_client:
            try:
                client = SafeHttpClientFactory._create_optimized_azure_client(azure_deployment, api_version, **kwargs)
                logger.info("Created Azure OpenAI client with optimized configuration")
                return client
            except Exception as e:
                logger.warning(f"Optimized Azure client failed: {e}, trying without custom http_client")
        
        # Strategy 2: Try without custom httpx client
        try:
            client = SafeHttpClientFactory._create_simple_azure_client(azure_deployment, api_version, **kwargs)
            logger.info("Created Azure OpenAI client with simple configuration")
            return client
        except Exception as e:
            logger.error(f"All Azure client configurations failed: {e}")
            raise ValueError(f"Cannot initialize Azure OpenAI client: {e}")
    
    @staticmethod
    def _create_optimized_azure_client(azure_deployment: str, api_version: str, **kwargs) -> Any:
        """Create Azure client with custom httpx configuration"""
        try:
            # Check if httpx is available and get version
            import httpx
            from openai import AzureOpenAI
            
            httpx_version = getattr(httpx, '__version__', '0.0.0')
            logger.debug(f"Using httpx version: {httpx_version}")
            
            # Version-compatible httpx configuration
            limits_config = SafeHttpClientFactory._get_safe_limits_config(httpx_version)
            timeout_config = SafeHttpClientFactory._get_safe_timeout_config(httpx_version)
            additional_config = SafeHttpClientFactory._get_additional_httpx_config(httpx_version)
            
            safe_http_client = httpx.Client(
                limits=limits_config,
                timeout=timeout_config,
                **additional_config
            )
            
            # Try to create Azure client with http_client parameter
            client_kwargs = {
                'azure_deployment': azure_deployment,
                'api_version': api_version,
                'http_client': safe_http_client,
                'max_retries': 3,
                **kwargs
            }
            
            return AzureOpenAI(**client_kwargs)
            
        except TypeError as e:
            if 'http_client' in str(e):
                logger.warning("AzureOpenAI does not support http_client parameter in this version")
                raise
            else:
                logger.error(f"Azure client creation failed: {e}")
                raise
        except Exception as e:
            logger.error(f"Optimized Azure client creation failed: {e}")
            raise
    
    @staticmethod
    def _create_simple_azure_client(azure_deployment: str, api_version: str, **kwargs) -> Any:
        """Create basic Azure client without custom httpx configuration"""
        try:
            from openai import AzureOpenAI
            
            # Try with timeout and max_retries if supported
            try:
                return AzureOpenAI(
                    azure_deployment=azure_deployment,
                    api_version=api_version,
                    timeout=30.0,
                    max_retries=2,
                    **kwargs
                )
            except TypeError:
                # If timeout/max_retries not supported, use basic client
                logger.info("Using basic Azure OpenAI client (timeout/max_retries not supported)")
                return AzureOpenAI(
                    azure_deployment=azure_deployment,
                    api_version=api_version,
                    **kwargs
                )
        except Exception as e:
            logger.error(f"Simple Azure client creation failed: {e}")
            raise
    
    @staticmethod
    def create_openrouter_client(api_key: str, base_url: str, default_headers: dict, use_custom_http_client: bool = True, **kwargs) -> Any:
        """
        Create OpenRouter client with safe configuration and version compatibility.
        
        Args:
            api_key: OpenRouter API key
            base_url: OpenRouter base URL
            default_headers: OpenRouter headers
            use_custom_http_client: Whether to try using custom httpx client
            **kwargs: Additional arguments to pass to OpenAI client
            
        Returns:
            Configured OpenAI client instance for OpenRouter
            
        Raises:
            ValueError: If client cannot be created with any configuration
        """
        
        # Strategy 1: Try with custom httpx client if requested
        if use_custom_http_client:
            try:
                client = SafeHttpClientFactory._create_optimized_openrouter_client(api_key, base_url, default_headers, **kwargs)
                logger.info("Created OpenRouter client with optimized configuration")
                return client
            except Exception as e:
                logger.warning(f"Optimized OpenRouter client failed: {e}, trying without custom http_client")
        
        # Strategy 2: Try without custom httpx client
        try:
            client = SafeHttpClientFactory._create_simple_openrouter_client(api_key, base_url, default_headers, **kwargs)
            logger.info("Created OpenRouter client with simple configuration")
            return client
        except Exception as e:
            logger.error(f"All OpenRouter client configurations failed: {e}")
            raise ValueError(f"Cannot initialize OpenRouter client: {e}")
    
    @staticmethod
    def _create_optimized_openrouter_client(api_key: str, base_url: str, default_headers: dict, **kwargs) -> Any:
        """Create OpenRouter client with custom httpx configuration"""
        try:
            # Check if httpx is available and get version
            import httpx
            from openai import OpenAI
            
            httpx_version = getattr(httpx, '__version__', '0.0.0')
            logger.debug(f"Using httpx version: {httpx_version}")
            
            # Version-compatible httpx configuration
            limits_config = SafeHttpClientFactory._get_safe_limits_config(httpx_version)
            timeout_config = SafeHttpClientFactory._get_safe_timeout_config(httpx_version)
            additional_config = SafeHttpClientFactory._get_additional_httpx_config(httpx_version)
            
            safe_http_client = httpx.Client(
                limits=limits_config,
                timeout=timeout_config,
                **additional_config
            )
            
            # Try to create OpenRouter client with http_client parameter
            client_kwargs = {
                'api_key': api_key,
                'base_url': base_url,
                'default_headers': default_headers,
                'http_client': safe_http_client,
                'max_retries': 3,
                **kwargs
            }
            
            return OpenAI(**client_kwargs)
            
        except TypeError as e:
            if 'http_client' in str(e):
                logger.warning("OpenAI client does not support http_client parameter in this version")
                raise
            else:
                logger.error(f"OpenRouter client creation failed: {e}")
                raise
        except Exception as e:
            logger.error(f"Optimized OpenRouter client creation failed: {e}")
            raise
    
    @staticmethod
    def _create_simple_openrouter_client(api_key: str, base_url: str, default_headers: dict, **kwargs) -> Any:
        """Create basic OpenRouter client without custom httpx configuration"""
        try:
            from openai import OpenAI
            
            # Try with timeout and max_retries if supported
            try:
                return OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    default_headers=default_headers,
                    timeout=30.0,
                    max_retries=2,
                    **kwargs
                )
            except TypeError:
                # If timeout/max_retries not supported, use basic client
                logger.info("Using basic OpenAI client for OpenRouter (timeout/max_retries not supported)")
                return OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    default_headers=default_headers,
                    **kwargs
                )
        except Exception as e:
            logger.error(f"Simple OpenRouter client creation failed: {e}")
            raise
    
    @staticmethod
    def create_genai_client(**kwargs) -> Any:
        """
        Create Google GenAI client with safe configuration.
        
        Args:
            **kwargs: Arguments to pass to GenAI client
            
        Returns:
            Configured GenAI client instance
            
        Raises:
            ValueError: If client cannot be created
        """
        try:
            client = SafeHttpClientFactory._create_optimized_genai_client(**kwargs)
            logger.info("Created GenAI client with optimized configuration")
            return client
        except Exception as e:
            logger.warning(f"Optimized GenAI client failed: {e}, trying fallback")
            client = SafeHttpClientFactory._create_fallback_genai_client(**kwargs) 
            logger.info("Created GenAI client with fallback configuration")
            return client
    
    @staticmethod
    def _create_optimized_genai_client(**kwargs) -> Any:
        """Create GenAI client with advanced configuration"""
        try:
            import google.genai as genai
            
            # Version detection
            genai_version = SafeHttpClientFactory._get_package_version('google-genai')
            logger.debug(f"Using google-genai version: {genai_version}")
            
            # Try to import types for advanced configuration
            try:
                from google.genai import types
                
                # Build http_options with version compatibility
                http_options_config = {
                    'timeout': 60_000,  # 60 seconds in milliseconds
                }
                
                # Add version-specific configurations
                try:
                    http_options_config['api_version'] = 'v1'
                    
                    # Try to add async_client_args (may not be available in all versions)
                    try:
                        import httpx
                        # Create custom httpx client with connection pooling limits
                        custom_limits = httpx.Limits(
                            max_connections=50,  # Reduced from 200 to prevent exhaustion
                            max_keepalive_connections=20,  # Reduced from 100
                            keepalive_expiry=30.0  # Close idle connections after 30s
                        )
                        http_options_config['async_client_args'] = {
                            'limits': custom_limits,
                            'timeout': httpx.Timeout(60.0, connect=10.0)
                        }
                    except Exception as e:
                        logger.debug(f"Could not set async_client_args: {e}")
                    
                except Exception as e:
                    logger.debug(f"Could not set advanced GenAI options: {e}")
                
                http_options = types.HttpOptions(**http_options_config)
                
                # Create client with configuration
                client_config = {**kwargs, 'http_options': http_options}
                return genai.Client(**client_config)
                
            except ImportError:
                logger.warning("google.genai.types not available, using basic client")
                # Fallback to basic client without advanced options
                return genai.Client(**kwargs)
            
        except ImportError as e:
            logger.warning(f"GenAI packages not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Optimized GenAI client creation failed: {e}")
            raise
    
    @staticmethod
    def _create_fallback_genai_client(**kwargs) -> Any:
        """Create basic GenAI client without advanced configuration"""
        try:
            import google.genai as genai
            return genai.Client(**kwargs)
        except Exception as e:
            logger.error(f"Fallback GenAI client creation failed: {e}")
            raise
    
    @staticmethod
    def _get_package_version(package_name: str) -> str:
        """Safely get package version"""
        try:
            # Try modern approach first
            import importlib.metadata
            return importlib.metadata.version(package_name)
        except ImportError:
            # Fallback for older Python versions
            try:
                import pkg_resources
                return pkg_resources.get_distribution(package_name).version
            except Exception:
                return "unknown"
        except Exception:
            try:
                # Alternative approach
                import pkg_resources
                return pkg_resources.get_distribution(package_name).version
            except Exception:
                return "unknown"
    
    @staticmethod
    def _create_standard_openai_client(api_key: str, **kwargs) -> Any:
        """Standard configuration with moderate limits"""
        try:
            import httpx
            from openai import OpenAI
            
            safe_http_client = httpx.Client(
                limits=httpx.Limits(
                    max_connections=50, 
                    max_keepalive_connections=25
                ),
                timeout=httpx.Timeout(30.0),
                verify=True  # SSL verification enabled for security
            )
            
            return OpenAI(
                api_key=api_key, 
                http_client=safe_http_client,
                max_retries=2,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Standard client creation failed: {e}")
            raise
    
    @staticmethod
    def _create_minimal_openai_client(api_key: str, **kwargs) -> Any:
        """Minimal configuration - just basic timeout"""
        try:
            from openai import OpenAI
            return OpenAI(
                api_key=api_key, 
                timeout=30.0,
                max_retries=1,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Minimal client creation failed: {e}")
            raise


class VersionMonitor:
    """Monitor package versions and alert on significant changes"""
    
    @staticmethod
    def log_environment_info():
        """Log current package versions for debugging"""
        packages = ['httpx', 'openai', 'google-genai']
        
        logger.info("HTTP Client Environment Information:")
        for package in packages:
            version = SafeHttpClientFactory._get_package_version(package)
            logger.info(f"  {package}: {version}")
    
    @staticmethod
    def check_compatibility():
        """Check if current package versions are known to work"""
        try:
            # Check httpx version
            import httpx
            httpx_version = getattr(httpx, '__version__', '0.0.0')
            major_version = int(httpx_version.split('.')[0])
            
            if major_version < 0 or major_version > 2:
                logger.warning(f"httpx version {httpx_version} may not be fully tested")
            
            # Check OpenAI version
            openai_version = SafeHttpClientFactory._get_package_version('openai')
            try:
                openai_major = int(openai_version.split('.')[0])
                if openai_major < 1 or openai_major > 2:
                    logger.warning(f"OpenAI version {openai_version} may not be fully tested")
            except Exception:
                pass
            
            # Check Google GenAI version
            genai_version = SafeHttpClientFactory._get_package_version('google-genai')
            if genai_version == "unknown":
                logger.info("google-genai package not installed or version unknown")
            
        except Exception as e:
            logger.warning(f"Version compatibility check failed: {e}")
    
    @staticmethod 
    def validate_configuration():
        """Validate that HTTP client configurations work"""
        try:
            # Test httpx configuration
            import httpx
            test_limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
            test_timeout = httpx.Timeout(5.0)
            
            # Create test client (don't use it, just validate configuration)
            test_client = httpx.Client(limits=test_limits, timeout=test_timeout)
            test_client.close()
            
            logger.info("HTTP client configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"HTTP client configuration validation failed: {e}")
            return False
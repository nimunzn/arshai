# Arshai Configuration System

## Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ARSHAI_ENV` | String | `development` | Environment name (development, staging, production) |
| `ARSHAI_LOG_LEVEL` | String | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `ARSHAI_CONFIG_PATH` | String | `./config.yaml` | Path to configuration file |
| `ARSHAI_OPENAI_API_KEY` | String | None | OpenAI API key for LLM access |
| `ARSHAI_AZURE_OPENAI_API_KEY` | String | None | Azure OpenAI API key (if using Azure) |
| `ARSHAI_AZURE_OPENAI_ENDPOINT` | String | None | Azure OpenAI endpoint URL |
| `ARSHAI_DEFAULT_MODEL` | String | `gpt-4o` | Default LLM model to use |
| `ARSHAI_EMBEDDING_MODEL` | String | `text-embedding-3-small` | Default embedding model |
| `ARSHAI_REDIS_URL` | String | None | Redis connection URL for memory storage |
| `ARSHAI_DB_URL` | String | None | Database connection URL |
| `ARSHAI_VECTOR_DB_URL` | String | None | Vector database connection URL |
| `ARSHAI_VECTOR_DB_TYPE` | String | `milvus` | Vector database type (milvus, pinecone, qdrant) |
| `ARSHAI_MEMORY_PROVIDER` | String | `in_memory` | Memory provider type (in_memory, redis, database) |
| `ARSHAI_CACHE_TTL` | Integer | `3600` | Cache time-to-live in seconds |
| `ARSHAI_WEBSEARCH_API_KEY` | String | None | API key for web search tools |
| `ARSHAI_MEMORY_WINDOW_SIZE` | Integer | `8192` | Default memory context window size |
| `ARSHAI_USE_SUMMARIZATION` | Boolean | `false` | Enable conversation summarization |
| `ARSHAI_API_HOST` | String | `0.0.0.0` | API server host |
| `ARSHAI_API_PORT` | Integer | `8000` | API server port |
| `ARSHAI_API_KEY` | String | None | API key for service authentication |
| `ARSHAI_DEFAULT_CHUNK_SIZE` | Integer | `512` | Default document chunk size |
| `ARSHAI_MAX_RETRY_ATTEMPTS` | Integer | `3` | Maximum API retry attempts |
| `ARSHAI_RETRY_BACKOFF_FACTOR` | Float | `0.5` | Retry exponential backoff factor |

## Configuration File Format

Arshai supports YAML configuration files for comprehensive system configuration. Below is an example configuration structure:

```yaml
environment: development

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  output:
    console: true
    file: logs/arshai.log

llm:
  providers:
    openai:
      api_key: "${ARSHAI_OPENAI_API_KEY}"
      organization_id: null
      default_model: gpt-4o
      timeout: 30
      request_options:
        temperature: 0.7
        max_tokens: 1024
    azure:
      api_key: "${ARSHAI_AZURE_OPENAI_API_KEY}"
      endpoint: "${ARSHAI_AZURE_OPENAI_ENDPOINT}"
      api_version: "2023-12-01-preview"
      deployment_map:
        gpt-4o: "your-deployment-name"

embedding:
  provider: openai
  model: text-embedding-3-small
  dimensions: 1536
  batch_size: 100

memory:
  provider: in_memory  # in_memory, redis, database
  window_size: 8192
  use_summarization: false
  summarization_threshold: 32
  redis:
    url: "${ARSHAI_REDIS_URL}"
    prefix: "arshai:memory:"
    ttl: 86400  # 24 hours
  database:
    url: "${ARSHAI_DB_URL}"
    table_prefix: "memory_"

vector_db:
  provider: milvus  # milvus, pinecone, qdrant
  connection:
    url: "${ARSHAI_VECTOR_DB_URL}"
    api_key: "${ARSHAI_VECTOR_DB_API_KEY}"
  settings:
    consistency_level: "Strong"
    timeout: 10

document_processing:
  chunk_size: 512
  chunk_overlap: 128
  chunking_strategy: paragraph
  add_metadata: true
  supported_formats:
    - pdf
    - docx
    - txt
    - md
    - html

agents:
  default_agent: operator
  operator:
    model: gpt-4o
    temperature: 0.7
    system_message: "You are a helpful AI assistant that can use tools to accomplish tasks."
  researcher:
    model: gpt-4o
    temperature: 0.2
    system_message: "You are a research assistant that prioritizes accuracy and sources information carefully."
  coder:
    model: gpt-4o
    temperature: 0.2
    system_message: "You are a coding assistant that helps write and improve code."

tools:
  enabled:
    - websearch
    - calculator
    - weather
    - file_io
  websearch:
    api_key: "${ARSHAI_WEBSEARCH_API_KEY}"
    search_engine: "google"
    result_count: 5

api:
  host: "${ARSHAI_API_HOST}"
  port: "${ARSHAI_API_PORT}"
  api_key: "${ARSHAI_API_KEY}"
  cors:
    allowed_origins:
      - "http://localhost:3000"
      - "https://yourdomain.com"
  rate_limit:
    enabled: true
    max_requests: 100
    time_window: 60  # seconds

workflows:
  max_nodes: 50
  execution_timeout: 300  # seconds
  persistence: true
  state_ttl: 3600  # seconds

speech:
  provider: openai
  default_voice: alloy
  max_input_size: 10485760  # 10MB
```

## Validation Rules and Constraints

### LLM Configuration
- API keys must be valid for the specified provider
- Model names must be supported by the provider
- Timeout must be positive and reasonable (1-120 seconds)
- Temperature must be between 0.0 and 2.0
- Token limits must respect provider maximums

### Memory Configuration
- Window size must be positive and respect model context limits
- Redis URL must be valid if Redis provider is selected
- Database URL must be valid if database provider is selected
- TTL values must be positive

### Vector Database Configuration
- Provider must be one of supported types
- Connection details must be valid for the selected provider
- Dimensions must match embedding model output
- Consistency level must be valid for the provider

### Document Processing Configuration
- Chunk size and overlap must be positive and reasonable
- Chunk size must be greater than overlap
- Supported formats must be in the allowed list
- Chunking strategy must be one of predefined strategies

### Tool Configuration
- API keys must be valid for the tools requiring them
- Tool-specific settings must respect tool requirements
- Enabled tools must exist in the system

### API Configuration
- Host and port must be valid network values
- CORS origins must be valid URLs
- Rate limit settings must be positive values

## Default Settings with Rationale

### LLM Defaults
- **Default Model**: `gpt-4o` - Provides good balance of capability and cost
- **Temperature**: `0.7` - Balanced between creativity and determinism
- **Max Tokens**: `1024` - Sufficient for most responses without excessive cost

### Memory Defaults
- **Provider**: `in_memory` - Simple setup without external dependencies
- **Window Size**: `8192` - Accommodates recent context for most LLMs
- **Summarization**: `disabled` - Reduces complexity for basic usage

### Document Processing Defaults
- **Chunk Size**: `512` - Balances context preservation with granularity
- **Chunking Strategy**: `paragraph` - Natural semantic boundaries
- **Add Metadata**: `true` - Preserves important context for retrieval

### API Defaults
- **Host**: `0.0.0.0` - Allows connections from any interface
- **Port**: `8000` - Common API port outside privileged range
- **Rate Limiting**: `enabled` - Provides basic protection against abuse

## Environment-specific Configurations

### Development Environment
```yaml
environment: development

logging:
  level: DEBUG
  output:
    console: true
    file: logs/arshai-dev.log

llm:
  providers:
    openai:
      default_model: gpt-3.5-turbo
      request_options:
        temperature: 1.0
        max_tokens: 2048

memory:
  provider: in_memory
  use_summarization: false

api:
  cors:
    allowed_origins:
      - "http://localhost:3000"
  rate_limit:
    enabled: false
```

### Staging Environment
```yaml
environment: staging

logging:
  level: INFO
  output:
    console: true
    file: logs/arshai-staging.log

llm:
  providers:
    openai:
      default_model: gpt-4o
      request_options:
        temperature: 0.7
        max_tokens: 1024

memory:
  provider: redis
  use_summarization: true

api:
  cors:
    allowed_origins:
      - "https://staging.yourdomain.com"
  rate_limit:
    enabled: true
    max_requests: 200
    time_window: 60
```

### Production Environment
```yaml
environment: production

logging:
  level: WARNING
  output:
    console: false
    file: logs/arshai-prod.log

llm:
  providers:
    openai:
      default_model: gpt-4o
      request_options:
        temperature: 0.5
        max_tokens: 1024

memory:
  provider: database
  use_summarization: true

api:
  cors:
    allowed_origins:
      - "https://yourdomain.com"
  rate_limit:
    enabled: true
    max_requests: 100
    time_window: 60
```

## Feature Flags and Toggles

### Feature Flag Configuration
```yaml
feature_flags:
  enable_speech_synthesis: true
  enable_web_search: true
  enable_conversation_summarization: true
  enable_reranking: false
  enable_streaming_responses: true
  enable_tool_usage: true
  enable_document_indexing: true
  experimental_plugins: false
  enable_azure_openai: false
  enable_anthropic_models: false
```

### Dynamic Configuration
Arshai supports dynamic configuration updates through:
- Configuration API endpoints (admin access required)
- Database-backed configuration store
- Redis-based feature flag service
- Environment-specific configuration files

## Secrets Management

### Secure Secrets Handling
- Environment variables for sensitive credentials
- Support for credential providers:
  - AWS Secrets Manager
  - HashiCorp Vault
  - Azure Key Vault
- Local .env file support for development
- Secret rotation capabilities

### Sensitive Data Management
- API keys masked in logs
- Credentials never exposed in API responses
- Automatic secure wiping of sensitive data from memory
- Encryption of sensitive configuration at rest

## Configuration Validation

### Validation Process
1. Schema validation against JSON Schema definitions
2. Cross-field validation for interdependent settings
3. Connection testing for external services
4. Comprehensive error reporting

### Error Handling
- Descriptive validation errors with field references
- Fallback to defaults for non-critical errors
- Application startup blocking for critical misconfigurations
- Configuration testing mode for validation without starting services

### Diagnostic Tools
- Configuration inspection command-line tools
- Validation reporting with suggested fixes
- Environment variable resolution display
- Current effective configuration dumping 
# Web Search Module

## Overview
The Web Search module provides a standardized interface for integrating external search engines into the Arshai framework. It enables applications to perform web searches and retrieve structured results from various search providers. The module currently supports SearxNG as the primary search engine implementation.

## Architecture
```
web_search/
├── searxng.py              # SearxNG search provider implementation
└── __init__.py             # Module exports
```

## Implementation Guide

### Using the SearxNG Search Client

```python
from src.web_search.searxng import SearxNGClient, SearxNGConfig
from seedwork.interfaces.iwebsearch import IWebSearchResult

# Create search client configuration
config = SearxNGConfig(
    host="https://your-searxng-instance.com",
    language="en",
    timeout=10,
    verify_ssl=True
)

# Initialize the search client
search_client = SearxNGClient(config)

# Perform synchronous search
results = search_client.search(
    query="artificial intelligence advancements",
    num_results=5,
    engines=["google", "bing"],
    categories=["science"]
)

# Process search results
for result in results:
    print(f"Title: {result.title}")
    print(f"URL: {result.url}")
    print(f"Content: {result.content}")
    print(f"Engines: {result.engines}")
    print(f"Category: {result.category}")
    print("---")

# For asynchronous usage
async def perform_async_search():
    async_results = await search_client.asearch(
        query="artificial intelligence advancements",
        num_results=5
    )
    return async_results
```

### Customizing Search Parameters

```python
# Search with custom parameters
results = search_client.search(
    query="python programming",
    num_results=10,
    engines=["duckduckgo", "stackoverflow"],
    categories=["it"],
    time_range="year",
    safesearch=1
)
```

## Integration Points

### With Tools
Web search can be integrated as a tool for agents:

```python
from src.factories import tool_factory
from src.web_search.searxng import SearxNGConfig, SearxNGClient

# Create web search client
config = SearxNGConfig(
    host=os.environ.get("SEARX_INSTANCE"),
    language="en"
)
search_client = SearxNGClient(config)

# Create a web search tool
web_search_tool = tool_factory.create_tool(
    "web_search",
    search_client=search_client
)

# Use the tool in an agent
agent = agent_factory.create_agent(
    agent_type="conversation",
    tools=[web_search_tool]
)
```

### With Workflows
Web search can be incorporated into workflows:

```python
from src.workflows import WorkflowRunner, Node
from src.web_search.searxng import SearxNGClient, SearxNGConfig

# Create search client
search_client = SearxNGClient(SearxNGConfig(
    host=os.environ.get("SEARX_INSTANCE")
))

# Create a search node in a workflow
search_node = Node(
    id="web_search",
    function=lambda query, **kwargs: search_client.search(query, **kwargs),
    input_mapping={"query": "user_query"}
)

# Add the node to a workflow
workflow = WorkflowRunner([
    # Other nodes...
    search_node,
    # Subsequent nodes that process search results...
])
```

## Configuration
Configure web search through the settings system:

```python
from src.config import Settings

settings = Settings()
search_config = settings.get_web_search_config("searxng")

# Create a search client with config
search_client = SearxNGClient(search_config)
```

### Web Search Configuration in YAML

```yaml
# in config.yaml
web_search:
  searxng:
    host: "https://searx.example.com"
    language: "en"
    timeout: 10
    verify_ssl: true
    default_engines:
      - "google"
      - "bing"
      - "duckduckgo"
    default_categories:
      - "general"
``` 
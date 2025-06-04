# Clients Utilities

## Overview

The clients utilities directory provides common helper classes and functions that support various client implementations in the Arshai framework. These utilities focus on standardizing connections to external services and providing reusable code across different client modules.

## Components

### RedisClient

The `RedisClient` is a singleton wrapper for Redis connections that:

- Provides asynchronous Redis connections
- Manages connection lifecycle
- Handles JSON serialization/deserialization
- Simplifies key-value operations

```python
import os
from src.clients.utils.redis_client import RedisClient

# Set a value with optional expiration
await RedisClient.set_key(
    key="user:profile:1234",
    value={"name": "Jane Doe", "email": "jane@example.com"},
    expire=3600  # 1 hour in seconds
)

# Get a value (automatically deserializes JSON)
profile = await RedisClient.get_key("user:profile:1234")
print(f"User name: {profile['name']}")

# Get raw Redis client for advanced operations
redis_client = await RedisClient.get_client()
keys = await redis_client.keys("user:profile:*")

# Clean up connection when done
await RedisClient.close_client()
```

## Configuration

The RedisClient uses environment variables for configuration:

```
REDIS_URL=redis://[:password@]host[:port][/database]
```

Example `.env` file:

```
REDIS_URL=redis://:password123@localhost:6379/0
```

## Error Handling

The RedisClient handles common Redis connection scenarios:

- Connection errors through proper exception handling
- JSON parsing errors with graceful fallbacks
- Binary data handling with proper encoding detection

## Integration

This utility is primarily used by:

- `ChatHistoryClient` for conversation persistence
- `MemoryManagerService` for working memory storage
- Various caching mechanisms throughout the framework

## Best Practices

1. Always close the client when finished to release resources:
   ```python
   await RedisClient.close_client()
   ```

2. Use appropriate expiration times for keys:
   ```python
   # Cache for 5 minutes
   await RedisClient.set_key("cache:results", results, expire=300)
   ```

3. Handle connection errors appropriately:
   ```python
   try:
       value = await RedisClient.get_key("my_key")
   except Exception as e:
       logger.error(f"Redis connection error: {str(e)}")
       # Implement fallback strategy
   ``` 
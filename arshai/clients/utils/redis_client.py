import json
import os
import redis.asyncio as redis
from dotenv import load_dotenv

load_dotenv()


class RedisClient:
    _client = None

    @classmethod
    async def get_client(cls):
        if not cls._client:
            cls._client = redis.Redis.from_url(os.getenv("REDIS_URL"))
        return cls._client

    @classmethod
    async def close_client(cls):
        if cls._client:
            await cls._client.close()
            cls._client = None

    @classmethod
    async def set_key(cls, key, value, expire=None):
        client = await cls.get_client()
        await client.set(key, value, ex=expire)

    @classmethod
    async def get_key(cls, key):
        client = await cls.get_client()
        value = await client.get(key)
        if value is not None:
            # Check if value is binary data (bytes) and decode it first
            if isinstance(value, bytes):
                try:
                    # Try to decode bytes to string using UTF-8
                    value = value.decode('utf-8')
                except UnicodeDecodeError:
                    # If it's not valid UTF-8, keep the binary data
                    pass
            
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                # If the value is not valid JSON, return it as is
                return value
        return None


async def get_redis_client():
    return await RedisClient.get_client()

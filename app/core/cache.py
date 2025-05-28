from typing import Optional, Any
import aioredis
import json
from app.core.config import get_settings
import logging

settings = get_settings()
logger = logging.getLogger(__name__)

class RedisCache:
    def __init__(self):
        self.redis = None
        self.cache_enabled = settings.CACHE_ENABLED
        self.cache_ttl = settings.CACHE_TTL

    async def connect(self):
        if not self.redis and self.cache_enabled:
            try:
                self.redis = await aioredis.from_url(
                    settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True
                )
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {str(e)}")
                self.cache_enabled = False

    async def get(self, key: str) -> Optional[Any]:
        if not self.cache_enabled:
            return None
            
        try:
            await self.connect()
            data = await self.redis.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return None

    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        if not self.cache_enabled:
            return False
            
        try:
            await self.connect()
            await self.redis.set(
                key,
                json.dumps(value),
                ex=ttl or self.cache_ttl
            )
            return True
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")
            return False

    async def delete(self, key: str) -> bool:
        if not self.cache_enabled:
            return False
            
        try:
            await self.connect()
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {str(e)}")
            return False

cache = RedisCache() 
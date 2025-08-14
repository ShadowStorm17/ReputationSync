"""
Advanced caching service with sophisticated caching strategies.
Provides multi-level caching and intelligent cache management.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, Optional

import redis.asyncio as redis

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class CacheStrategy:
    """Base class for cache strategies."""

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        raise NotImplementedError

    async def set(
        self, key: str, value: Any, ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache."""
        raise NotImplementedError

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        raise NotImplementedError


class RedisStrategy(CacheStrategy):
    """Redis-based caching strategy."""

    def __init__(self):
        """Initialize Redis strategy."""
        self.redis = redis.Redis.from_url(
            settings.REDIS_URL, encoding="utf-8", decode_responses=True
        )

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        try:
            value = await self.redis.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Redis get error: {str(e)}")
            return None

    async def set(
        self, key: str, value: Any, ttl: Optional[int] = None
    ) -> bool:
        """Set value in Redis."""
        try:
            await self.redis.set(key, json.dumps(value), ex=ttl)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {str(e)}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from Redis."""
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis delete error: {str(e)}")
            return False


class MemoryStrategy(CacheStrategy):
    """In-memory caching strategy."""

    def __init__(self):
        """Initialize memory strategy."""
        self.cache: Dict[str, Dict[str, Any]] = {}

    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory."""
        try:
            cache_data = self.cache.get(key)
            if not cache_data:
                return None

            # Check expiration
            if "expires_at" in cache_data:
                if datetime.utcnow() > cache_data["expires_at"]:
                    del self.cache[key]
                    return None

            return cache_data["value"]

        except Exception as e:
            logger.error(f"Memory get error: {str(e)}")
            return None

    async def set(
        self, key: str, value: Any, ttl: Optional[int] = None
    ) -> bool:
        """Set value in memory."""
        try:
            cache_data = {"value": value, "created_at": datetime.utcnow()}

            if ttl:
                cache_data["expires_at"] = datetime.utcnow() + timedelta(
                    seconds=ttl
                )

            self.cache[key] = cache_data
            return True

        except Exception as e:
            logger.error(f"Memory set error: {str(e)}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from memory."""
        try:
            if key in self.cache:
                del self.cache[key]
            return True
        except Exception as e:
            logger.error(f"Memory delete error: {str(e)}")
            return False


class MultiLevelCache:
    """Multi-level caching system."""

    def __init__(self):
        """Initialize multi-level cache."""
        self.levels = [
            MemoryStrategy(),
            RedisStrategy(),
        ]  # L1 cache  # L2 cache

    async def get(
        self, key: str, level: Optional[int] = None
    ) -> Optional[Any]:
        """Get value from cache levels."""
        try:
            if level is not None:
                return await self.levels[level].get(key)

            # Try each level
            for i, cache in enumerate(self.levels):
                value = await cache.get(key)
                if value is not None:
                    # Propagate to higher levels
                    for j in range(i):
                        await self.levels[j].set(key, value)
                    return value

            return None

        except Exception as e:
            logger.error(f"Multi-level get error: {str(e)}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        level: Optional[int] = None,
    ) -> bool:
        """Set value in cache levels."""
        try:
            if level is not None:
                return await self.levels[level].set(key, value, ttl)

            # Set in all levels
            success = True
            for cache in self.levels:
                if not await cache.set(key, value, ttl):
                    success = False

            return success

        except Exception as e:
            logger.error(f"Multi-level set error: {str(e)}")
            return False

    async def delete(self, key: str, level: Optional[int] = None) -> bool:
        """Delete value from cache levels."""
        try:
            if level is not None:
                return await self.levels[level].delete(key)

            # Delete from all levels
            success = True
            for cache in self.levels:
                if not await cache.delete(key):
                    success = False

            return success

        except Exception as e:
            logger.error(f"Multi-level delete error: {str(e)}")
            return False


def cache_result(
    ttl: Optional[int] = None, key_prefix: str = "", use_args: bool = True
):
    """Cache decorator for function results."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_service = CacheService()

            # Generate cache key
            if use_args:
                key_parts = [
                    key_prefix,
                    func.__name__,
                    hashlib.md5(str(args).encode()).hexdigest(),
                    hashlib.md5(str(kwargs).encode()).hexdigest(),
                ]
            else:
                key_parts = [key_prefix, func.__name__]

            cache_key = ":".join(filter(None, key_parts))

            # Try to get from cache
            cached_result = await cache_service.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            await cache_service.set(cache_key, result, ttl)

            return result

        return wrapper

    return decorator


class CacheService:
    """Advanced caching service."""

    def __init__(self):
        """Initialize cache service."""
        self.cache = MultiLevelCache()
        self.default_ttl = 3600  # 1 hour

    async def get(
        self, key: str, level: Optional[int] = None
    ) -> Optional[Any]:
        """Get value from cache."""
        return await self.cache.get(key, level)

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        level: Optional[int] = None,
    ) -> bool:
        """Set value in cache."""
        return await self.cache.set(key, value, ttl or self.default_ttl, level)

    async def delete(self, key: str, level: Optional[int] = None) -> bool:
        """Delete value from cache."""
        return await self.cache.delete(key, level)

    async def clear_pattern(self, pattern: str) -> bool:
        """Clear all keys matching pattern."""
        try:
            redis_strategy = self.cache.levels[1]
            keys = await redis_strategy.redis.keys(pattern)

            if keys:
                # Clear from all levels
                for key in keys:
                    await self.delete(key)

            return True

        except Exception as e:
            logger.error(f"Pattern clear error: {str(e)}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            redis_strategy = self.cache.levels[1]
            memory_strategy = self.cache.levels[0]

            stats = {
                "memory_keys": len(memory_strategy.cache),
                "redis_keys": await redis_strategy.redis.dbsize(),
                "memory_usage": sum(
                    len(str(v).encode())
                    for v in memory_strategy.cache.values()
                ),
                "redis_info": await redis_strategy.redis.info(),
            }

            return {"status": "success", "stats": stats}

        except Exception as e:
            logger.error(f"Stats error: {str(e)}")
            return {"status": "error", "message": str(e)}

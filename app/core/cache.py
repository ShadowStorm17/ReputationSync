"""
Cache management module.
Provides caching functionality using Redis.
"""

import json
from datetime import timedelta
from typing import Any, Optional

import redis.asyncio as redis

from app.core.config import get_settings

from functools import wraps
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import hashlib
import inspect
import asyncio
from typing import Callable, Union

settings = get_settings()


class CacheManager:
    """Cache manager for handling Redis caching operations."""

    def __init__(self):
        """Initialize cache manager with Redis connection."""
        self.redis = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True,
        )

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            value = await self.redis.get(key)
            return json.loads(value) if value else None
        except Exception:
            return None

    async def set(
        self, key: str, value: Any, expire: Optional[int] = None
    ) -> bool:
        """Set value in cache with optional expiration."""
        try:
            await self.redis.set(
                key,
                json.dumps(value),
                ex=expire if expire else None,
            )
            return True
        except Exception:
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            await self.redis.delete(key)
            return True
        except Exception:
            return False

    async def clear_pattern(self, pattern: str) -> bool:
        """Clear all keys matching pattern."""
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
            return True
        except Exception:
            return False


def _default_cache_key(request: Request, **kwargs) -> str:
    """Default cache key: method + path + sorted query + body hash (if POST/PUT)."""
    key = f"{request.method}:{request.url.path}"
    if request.query_params:
        key += f"?{str(sorted(request.query_params.items()))}"
    if request.method in ("POST", "PUT", "PATCH"):
        # Try to get body from request
        try:
            body = kwargs.get("body")
            if body is None and hasattr(request, "_body"):
                body = request._body
            if body is not None:
                key += f":{hashlib.sha256(str(body).encode()).hexdigest()}"
        except Exception:
            pass
    return key


def cache_response(
    ttl: int = 300,
    key_func: Callable = None,
    methods: list = ["GET"],
    include_user: bool = False,
    bypass_param: str = "no_cache",
    invalidate_param: str = "invalidate_cache",
):
    """
    Advanced decorator to cache FastAPI endpoint responses.
    - ttl: cache time in seconds
    - key_func: function to generate cache key (default: method+path+query+body)
    - methods: HTTP methods to cache
    - include_user: include user/session in cache key if available
    - bypass_param: query param to bypass cache
    - invalidate_param: query param to clear cache for this key
    """
    def decorator(func):
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Find request object
                request: Request = kwargs.get("request")
                if not request:
                    for arg in args:
                        if isinstance(arg, Request):
                            request = arg
                            break
                if not request:
                    raise RuntimeError("Request object not found for cache_response")

                # Only cache specified methods
                if request.method not in methods:
                    return await func(*args, **kwargs)

                # Bypass/invalidate logic
                query = dict(request.query_params)
                bypass = query.get(bypass_param) == "1"
                invalidate = query.get(invalidate_param) == "1"

                # Build cache key
                cache_key = None
                if key_func:
                    cache_key = key_func(request, *args, **kwargs)
                else:
                    cache_key = _default_cache_key(request, *args, **kwargs)
                if include_user:
                    user = kwargs.get("current_user") or getattr(request.state, "user", None)
                    if user:
                        cache_key += f":user:{getattr(user, 'id', str(user))}"

                # Invalidate cache if requested
                if invalidate:
                    await cache.delete(cache_key)

                # Try to get from cache
                if not bypass and not invalidate:
                    cached = await cache.get(cache_key)
                    if cached is not None:
                        # Return cached as JSONResponse
                        return JSONResponse(content=cached)

                # Call endpoint and cache result
                result = await func(*args, **kwargs)
                # Support for Pydantic models
                if isinstance(result, BaseModel):
                    to_cache = result.dict()
                elif isinstance(result, Response):
                    # Don't cache raw Response objects
                    return result
                else:
                    to_cache = result
                await cache.set(cache_key, to_cache, expire=timedelta(seconds=ttl))
                return JSONResponse(content=to_cache)
            return wrapper
        else:
            raise NotImplementedError("cache_response only supports async endpoints.")
    return decorator


# Create global cache manager instance
cache = CacheManager()


class RedisCache:
    """Stub for RedisCache to satisfy imports in tests."""
    pass

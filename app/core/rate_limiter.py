"""
Rate limiter module.
Handles request rate limiting with distributed support.
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from app.core.config import get_settings
from app.core.error_handling import (ErrorCategory, ErrorSeverity,
                                     ReputationError)
from app.core.logging import logger

settings = get_settings()


class RateLimiter:
    """Rate limiter with distributed support."""

    def __init__(self) -> None:
        """Initialize rate limiter."""
        self._redis = None
        self._local_cache: Dict[str, List[float]] = {}
        self._local_lock = asyncio.Lock()
        self._cleanup_interval = 3600  # 1 hour
        self._last_cleanup = time.time()
        self._max_retries = 3
        self._retry_delay = 1  # seconds
        self._redis_connection_timeout = 5  # seconds
        self._redis_operation_timeout = 2  # seconds
        self._initialize_redis()

    def _initialize_redis(self) -> None:
        """Initialize Redis connection."""
        try:
            import redis
            self._redis = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            # Test connection
            self._redis.ping()
            logger.info("Redis connection established")
        except ImportError:
            logger.warning("Redis not available, using local cache only")
            self._redis = None
        except Exception as e:
            logger.error(f"Redis connection failed: {str(e)}")
            self._redis = None

    async def check_rate_limit(
        self,
        key: str,
        max_requests: int,
        window: int
    ) -> bool:
        """Check if request is within rate limit."""
        try:
            # Try Redis first
            if await self._check_redis_rate_limit(key, max_requests, window):
                return True

            # Fallback to local cache
            async with self._local_lock:
                now = time.time()

                # Cleanup old entries
                if now - self._last_cleanup > self._cleanup_interval:
                    await self._cleanup_local_cache()

                # Get request history
                history = self._local_cache.get(key, [])

                # Remove old requests
                history = [t for t in history if now - t < window]

                # Check if limit exceeded
                if len(history) >= max_requests:
                    return False

                # Add current request
                history.append(now)
                self._local_cache[key] = history

                return True

        except Exception as e:
            logger.error(f"Rate limit check error: {str(e)}", exc_info=True)
            return True  # Allow request on error

    async def _check_redis_rate_limit(
        self,
        key: str,
        max_requests: int,
        window: int
    ) -> bool:
        """Check rate limit using Redis."""
        if not self._redis:
            return False

        for attempt in range(self._max_retries):
            try:
                # Use Redis pipeline for atomic operations
                pipe = self._redis.pipeline()
                now = time.time()

                # Add current request
                pipe.zadd(key, {str(now): now})

                # Remove old requests
                pipe.zremrangebyscore(key, 0, now - window)

                # Count requests in window
                pipe.zcard(key)

                # Set expiry
                pipe.expire(key, window)

                # Execute pipeline
                _, _, request_count, _ = pipe.execute()

                return request_count <= max_requests

            except Exception as e:
                logger.error(
                    f"Redis rate limit error (attempt {attempt + 1}/{self._max_retries}): {str(e)}",
                    exc_info=True
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay)
                    continue
                return True  # Allow request on Redis error

    async def _cleanup_local_cache(self) -> None:
        """Clean up expired entries from local cache."""
        try:
            now = time.time()
            expired_keys = []

            for key, history in self._local_cache.items():
                # Remove old requests
                self._local_cache[key] = [
                    t for t in history
                    if now - t < 3600  # 1 hour
                ]

                # Remove empty history
                if not self._local_cache[key]:
                    expired_keys.append(key)

            # Remove expired keys
            for key in expired_keys:
                del self._local_cache[key]

            self._last_cleanup = now

        except Exception as e:
            logger.error(f"Local cache cleanup error: {str(e)}", exc_info=True)

    async def get_rate_limit_info(self, key: str) -> Dict[str, Any]:
        """Get rate limit information."""
        try:
            info: Dict[str, Any] = {
                "redis_count": 0,
                "redis_ttl": 0,
                "local_count": 0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            # Get Redis info
            if self._redis:
                try:
                    info["redis_count"] = self._redis.zcard(key)
                    info["redis_ttl"] = self._redis.ttl(key)
                except Exception as e:
                    logger.error(f"Redis info error: {str(e)}", exc_info=True)

            # Get local cache info
            async with self._local_lock:
                info["local_count"] = len(self._local_cache.get(key, []))

            return info

        except Exception as e:
            logger.error(f"Rate limit info error: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def reset_rate_limit(self, key: str) -> None:
        """Reset rate limit for key."""
        try:
            # Reset Redis
            if self._redis:
                try:
                    self._redis.delete(key)
                except Exception as e:
                    logger.error(f"Redis reset error: {str(e)}", exc_info=True)

            # Reset local cache
            async with self._local_lock:
                if key in self._local_cache:
                    del self._local_cache[key]

        except Exception as e:
            logger.error(f"Rate limit reset error: {str(e)}", exc_info=True)
            raise ReputationError(
                message=f"Failed to reset rate limit: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.SYSTEM
            )

    async def is_rate_limited(self, key: str, max_requests: int, window: int) -> bool:
        """Return True if the rate limit is exceeded, False otherwise."""
        allowed = await self.check_rate_limit(key, max_requests, window)
        return not allowed


rate_limiter = RateLimiter()

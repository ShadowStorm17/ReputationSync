"""
Rate limiting service for request throttling.
Provides advanced rate limiting capabilities.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import redis.asyncio as redis

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RateLimitStrategy:
    """Base class for rate limit strategies."""

    async def check_limit(
        self, key: str, limit: int, window: int
    ) -> Dict[str, Any]:
        """Check rate limit."""
        raise NotImplementedError


class FixedWindowStrategy(RateLimitStrategy):
    """Fixed window rate limiting."""

    def __init__(self):
        """Initialize fixed window strategy."""
        self.redis = redis.Redis.from_url(
            settings.REDIS_URL, encoding="utf-8", decode_responses=True
        )

    async def check_limit(
        self, key: str, limit: int, window: int
    ) -> Dict[str, Any]:
        """Check rate limit using fixed window."""
        try:
            # Get current window
            window_key = f"{key}:{int(datetime.utcnow().timestamp() / window)}"

            # Get current count
            count = await self.redis.get(window_key)
            current = int(count) if count else 0

            if current >= limit:
                ttl = await self.redis.ttl(window_key)
                return {"allowed": False, "remaining": 0, "reset_after": ttl}

            # Increment counter
            pipe = self.redis.pipeline()
            await pipe.incr(window_key)
            await pipe.expire(window_key, window)
            await pipe.execute()

            return {
                "allowed": True,
                "remaining": limit - (current + 1),
                "reset_after": window,
            }

        except Exception as e:
            logger.error(f"Fixed window check error: {str(e)}")
            return {"allowed": False, "remaining": 0, "reset_after": 0}


class SlidingWindowStrategy(RateLimitStrategy):
    """Sliding window rate limiting."""

    def __init__(self):
        """Initialize sliding window strategy."""
        self.redis = redis.Redis.from_url(
            settings.REDIS_URL, encoding="utf-8", decode_responses=True
        )

    async def check_limit(
        self, key: str, limit: int, window: int
    ) -> Dict[str, Any]:
        """Check rate limit using sliding window."""
        try:
            now = datetime.utcnow().timestamp()
            window_start = now - window

            # Remove old entries
            await self.redis.zremrangebyscore(key, "-inf", window_start)

            # Get current count
            count = await self.redis.zcard(key)

            if count >= limit:
                # Get reset time
                oldest = await self.redis.zrange(key, 0, 0, withscores=True)
                reset_after = (
                    int(oldest[0][1] + window - now) if oldest else window
                )

                return {
                    "allowed": False,
                    "remaining": 0,
                    "reset_after": reset_after,
                }

            # Add new entry
            await self.redis.zadd(key, {str(now): now})

            # Set expiry
            await self.redis.expire(key, window)

            return {
                "allowed": True,
                "remaining": limit - (count + 1),
                "reset_after": window,
            }

        except Exception as e:
            logger.error(f"Sliding window check error: {str(e)}")
            return {"allowed": False, "remaining": 0, "reset_after": 0}


class TokenBucketStrategy(RateLimitStrategy):
    """Token bucket rate limiting."""

    def __init__(self):
        """Initialize token bucket strategy."""
        self.redis = redis.Redis.from_url(
            settings.REDIS_URL, encoding="utf-8", decode_responses=True
        )

    async def check_limit(
        self, key: str, limit: int, window: int
    ) -> Dict[str, Any]:
        """Check rate limit using token bucket."""
        try:
            now = datetime.utcnow().timestamp()

            # Get bucket data
            bucket_key = f"bucket:{key}"
            bucket_data = await self.redis.get(bucket_key)

            if bucket_data:
                bucket = json.loads(bucket_data)
                tokens = bucket["tokens"]
                last_refill = bucket["last_refill"]
            else:
                tokens = limit
                last_refill = now

            # Calculate token refill
            refill_rate = limit / window
            elapsed = now - last_refill
            refill = elapsed * refill_rate

            # Update tokens
            tokens = min(limit, tokens + refill)

            if tokens < 1:
                # Calculate reset time
                reset_after = int((1 - tokens) / refill_rate)

                # Store updated bucket
                await self.redis.set(
                    bucket_key,
                    json.dumps({"tokens": tokens, "last_refill": now}),
                    ex=window,
                )

                return {
                    "allowed": False,
                    "remaining": 0,
                    "reset_after": reset_after,
                }

            # Consume token
            tokens -= 1

            # Store updated bucket
            await self.redis.set(
                bucket_key,
                json.dumps({"tokens": tokens, "last_refill": now}),
                ex=window,
            )

            return {
                "allowed": True,
                "remaining": int(tokens),
                "reset_after": int((1 - tokens % 1) / refill_rate)
                if tokens < limit
                else 0,
            }

        except Exception as e:
            logger.error(f"Token bucket check error: {str(e)}")
            return {"allowed": False, "remaining": 0, "reset_after": 0}


class RateLimitService:
    """Advanced rate limiting service."""

    def __init__(self):
        """Initialize rate limit service."""
        self.strategies = {
            "fixed": FixedWindowStrategy(),
            "sliding": SlidingWindowStrategy(),
            "token": TokenBucketStrategy(),
        }
        self.default_strategy = "sliding"

    async def check_rate_limit(
        self, key: str, limit: int, window: int, strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check rate limit."""
        try:
            # Get strategy
            strategy_name = strategy or self.default_strategy
            limiter = self.strategies.get(strategy_name)

            if not limiter:
                return {
                    "status": "error",
                    "message": f"Unknown strategy: {strategy_name}",
                }

            # Check limit
            result = await limiter.check_limit(key, limit, window)

            return {
                "status": "success",
                "allowed": result["allowed"],
                "remaining": result["remaining"],
                "reset_after": result["reset_after"],
            }

        except Exception as e:
            logger.error(f"Rate limit check error: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def get_limit_status(
        self, key: str, strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get current rate limit status."""
        try:
            # Get strategy
            strategy_name = strategy or self.default_strategy
            limiter = self.strategies.get(strategy_name)

            if not limiter:
                return {
                    "status": "error",
                    "message": f"Unknown strategy: {strategy_name}",
                }

            # Get Redis client
            redis = limiter.redis

            if strategy_name == "fixed":
                window_key = f"{key}:{int(datetime.utcnow().timestamp())}"
                count = await redis.get(window_key)
                ttl = await redis.ttl(window_key)

                return {
                    "status": "success",
                    "count": int(count) if count else 0,
                    "ttl": ttl,
                }

            elif strategy_name == "sliding":
                now = datetime.utcnow().timestamp()
                count = await redis.zcard(key)
                oldest = await redis.zrange(key, 0, 0, withscores=True)

                return {
                    "status": "success",
                    "count": count,
                    "window_start": oldest[0][1] if oldest else now,
                }

            elif strategy_name == "token":
                bucket_key = f"bucket:{key}"
                bucket_data = await redis.get(bucket_key)

                if bucket_data:
                    bucket = json.loads(bucket_data)
                    return {
                        "status": "success",
                        "tokens": bucket["tokens"],
                        "last_refill": bucket["last_refill"],
                    }

                return {"status": "success", "tokens": 0, "last_refill": 0}

        except Exception as e:
            logger.error(f"Status check error: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def reset_limits(
        self, key: str, strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """Reset rate limits for key."""
        try:
            # Get strategy
            strategy_name = strategy or self.default_strategy
            limiter = self.strategies.get(strategy_name)

            if not limiter:
                return {
                    "status": "error",
                    "message": f"Unknown strategy: {strategy_name}",
                }

            # Get Redis client
            redis = limiter.redis

            if strategy_name == "fixed":
                pattern = f"{key}:*"
                keys = await redis.keys(pattern)
                if keys:
                    await redis.delete(*keys)

            elif strategy_name == "sliding":
                await redis.delete(key)

            elif strategy_name == "token":
                bucket_key = f"bucket:{key}"
                await redis.delete(bucket_key)

            return {"status": "success", "message": "Rate limits reset"}

        except Exception as e:
            logger.error(f"Reset error: {str(e)}")
            return {"status": "error", "message": str(e)}

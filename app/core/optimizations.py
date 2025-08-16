"""
Optimization implementations for the API.
Provides performance, security, and reliability enhancements.
"""

from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, List, Optional

import redis.asyncio as redis
from prometheus_client import Counter, Gauge, Histogram

from app.core.cache import cache
from app.core.config import get_settings
from app.core.error_handling import SystemError

settings = get_settings()

# Performance metrics
REQUEST_DURATION = Histogram(
    "request_duration_seconds",
    "Request duration in seconds",
    ["endpoint", "method"],
)
CACHE_HITS = Counter("cache_hits_total", "Cache hits", ["cache_level"])
CACHE_MISSES = Counter("cache_misses_total", "Cache misses", ["cache_level"])
CIRCUIT_BREAKER_STATE = Gauge(
    "circuit_breaker_state", "Circuit breaker state", ["endpoint"]
)


class HierarchicalCache:
    """Implements a hierarchical caching system."""

    def __init__(self):
        """Initialize the hierarchical cache."""
        self.l1_cache = {}  # In-memory cache
        self.l2_cache = cache  # Redis cache

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache hierarchy."""
        # Try L1 cache
        if key in self.l1_cache:
            CACHE_HITS.labels(cache_level="l1").inc()
            return self.l1_cache[key]

        # Try L2 cache
        value = await self.l2_cache.get(key)
        if value is not None:
            CACHE_HITS.labels(cache_level="l2").inc()
            self.l1_cache[key] = value  # Promote to L1
            return value

        CACHE_MISSES.labels(cache_level="all").inc()
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache hierarchy."""
        self.l1_cache[key] = value
        await self.l2_cache.set(key, value, ttl)


class CircuitBreaker:
    """Implements the circuit breaker pattern."""

    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        """Initialize circuit breaker."""
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def __call__(self, func):
        """Decorator for circuit breaker pattern."""

        @wraps(func)
        async def wrapper(*args, **kwargs):
            await self.before_call()
            try:
                result = await func(*args, **kwargs)
                await self.on_success()
                return result
            except Exception as e:
                await self.on_failure()
                raise e

        return wrapper

    async def before_call(self):
        """Check circuit state before call."""
        if self.state == "open":
            if (
                self.last_failure_time
                and datetime.now(timezone.utc) - self.last_failure_time
                > timedelta(seconds=self.reset_timeout)
            ):
                self.state = "half-open"
            else:
                raise SystemError("Circuit breaker is open")

    async def on_success(self):
        """Handle successful call."""
        if self.state == "half-open":
            self.state = "closed"
            self.failures = 0
            CIRCUIT_BREAKER_STATE.labels(endpoint="all").set(0)

    async def on_failure(self):
        """Handle failed call."""
        self.failures += 1
        self.last_failure_time = datetime.now(timezone.utc)

        if self.failures >= self.failure_threshold:
            self.state = "open"
            CIRCUIT_BREAKER_STATE.labels(endpoint="all").set(1)


class RateLimiter:
    """Enhanced rate limiter with Redis backend."""

    def __init__(self, requests: int, window: int):
        """Initialize rate limiter."""
        self.requests = requests
        self.window = window
        self.redis = redis.from_url(settings.REDIS_URL)

    async def is_allowed(self, key: str) -> bool:
        """Check if request is allowed."""
        current = await self.redis.get(f"ratelimit:{key}")

        if not current:
            await self.redis.setex(f"ratelimit:{key}", self.window, 1)
            return True

        count = int(current)
        if count < self.requests:
            await self.redis.incr(f"ratelimit:{key}")
            return True

        return False


def cache_warmer(keys: List[str]):
    """Decorator for cache warming."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Warm up cache for specified keys
            cache = HierarchicalCache()
            for key in keys:
                if await cache.get(key) is None:
                    value = await func(*args, **kwargs)
                    await cache.set(key, value)
                    return value
            return await func(*args, **kwargs)

        return wrapper

    return decorator


class QueryOptimizer:
    """Optimizes database queries."""

    @staticmethod
    async def optimize_query(query: str) -> str:
        """Optimize a SQL query."""
        # Add query optimization logic here
        return query

    @staticmethod
    def should_use_materialized_view(query: str) -> bool:
        """Determine if query should use materialized view."""
        # Add materialized view logic here
        return False


class PerformanceMonitor:
    """Monitors API performance."""

    def __init__(self):
        """Initialize performance monitor."""
        self.slow_queries = []
        self.response_times = []

    async def record_query(self, query: str, duration: float):
        """Record a slow query."""
        if duration > settings.SLOW_QUERY_THRESHOLD:
            self.slow_queries.append(
                {
                    "query": query,
                    "duration": duration,
                    "timestamp": datetime.now(timezone.utc),
                }
            )

    async def record_response_time(self, endpoint: str, duration: float):
        """Record API response time."""
        self.response_times.append(
            {
                "endpoint": endpoint,
                "duration": duration,
                "timestamp": datetime.now(timezone.utc),
            }
        )
        REQUEST_DURATION.labels(endpoint=endpoint, method="GET").observe(
            duration
        )

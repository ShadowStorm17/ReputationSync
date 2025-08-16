"""
Performance optimization service.
Handles caching, connection pooling, and batch processing.
"""

import asyncio
import json
import logging
import hashlib
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

import redis.asyncio as redis

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

T = TypeVar('T')


class CacheManager:
    """Advanced caching system."""

    def __init__(self):
        """Initialize cache manager."""
        self.redis = redis.Redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        self.default_ttl = 3600  # 1 hour

    async def get(
        self,
        key: str
    ) -> Optional[Any]:
        """Get value from cache."""
        try:
            value = await self.redis.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error("Error getting from cache: %s", e)
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """Set value in cache."""
        try:
            await self.redis.set(
                key,
                json.dumps(value),
                ex=ttl or self.default_ttl
            )
        except Exception as e:
            logger.error("Error setting cache: %s", e)

    async def invalidate(
        self,
        pattern: str
    ):
        """Invalidate cache entries matching pattern."""
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
        except Exception as e:
            logger.error("Error invalidating cache: %s", e)


class ConnectionPool:
    """Database connection pooling system."""

    def __init__(
        self,
        min_size: int = 5,
        max_size: int = 20
    ):
        """Initialize connection pool."""
        self.min_size = min_size
        self.max_size = max_size
        self.pool = asyncio.Queue(maxsize=max_size)
        self.active_connections = 0

    async def get_connection(self):
        """Get connection from pool."""
        if self.pool.empty() and self.active_connections < self.max_size:
            # Create new connection
            connection = await self._create_connection()
            self.active_connections += 1
            return connection

        # Wait for available connection
        return await self.pool.get()

    async def release_connection(
        self,
        connection: Any
    ):
        """Release connection back to pool."""
        if self.pool.full():
            # Close excess connections
            await self._close_connection(connection)
            self.active_connections -= 1
        else:
            await self.pool.put(connection)

    async def _create_connection(self):
        """Create new database connection."""
        # Implement connection creation logic
        return {}

    async def _close_connection(
        self,
        connection: Any
    ):
        """Close database connection."""
        # Implement connection closing logic


class BatchProcessor:
    """Batch processing system."""

    def __init__(
        self,
        batch_size: int = 100,
        flush_interval: int = 60
    ):
        """Initialize batch processor."""
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.batches: Dict[str, List[Any]] = {}
        self.last_flush: Dict[str, datetime] = {}

    async def add_to_batch(
        self,
        batch_type: str,
        item: Any
    ):
        """Add item to batch."""
        if batch_type not in self.batches:
            self.batches[batch_type] = []
            self.last_flush[batch_type] = datetime.now(timezone.utc)

        self.batches[batch_type].append(item)

        # Check if batch should be processed
        if len(self.batches[batch_type]) >= self.batch_size:
            await self.process_batch(batch_type)
        elif (
            datetime.now(timezone.utc) - self.last_flush[batch_type]
        ).total_seconds() >= self.flush_interval:
            await self.process_batch(batch_type)

    async def process_batch(
        self,
        batch_type: str
    ):
        """Process batch of items."""
        if batch_type not in self.batches:
            return

        items = self.batches[batch_type]
        if not items:
            return

        try:
            # Process items based on batch type
            if batch_type == 'metrics':
                await self._process_metrics_batch(items)
            elif batch_type == 'events':
                await self._process_events_batch(items)

            # Clear processed items
            self.batches[batch_type] = []
            self.last_flush[batch_type] = datetime.now(timezone.utc)

        except Exception as e:
            logger.error("Error processing batch: %s", e)

    async def _process_metrics_batch(
        self,
        items: List[Any]
    ):
        """Process batch of metrics."""
        # Implement metrics batch processing

    async def _process_events_batch(
        self,
        items: List[Any]
    ):
        """Process batch of events."""
        # Implement events batch processing


def cache_result(
    ttl: Optional[int] = None,
    key_prefix: str = ""
):
    """Cache function result decorator."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_manager = CacheManager()

            # Generate cache key
            key_material = {
                "args": args,
                "kwargs": kwargs,
            }
            digest = hashlib.sha256(json.dumps(key_material, default=str, sort_keys=True).encode("utf-8")).hexdigest()
            cache_key = f"{key_prefix}:{func.__name__}:{digest}"

            # Try to get from cache
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            await cache_manager.set(cache_key, result, ttl)

            return result
        return wrapper
    return decorator


class OptimizationService:
    """Performance optimization service."""

    def __init__(self):
        """Initialize optimization service."""
        self.cache_manager = CacheManager()
        self.connection_pool = ConnectionPool()
        self.batch_processor = BatchProcessor()

    async def optimize_query(
        self,
        query: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize database query."""
        try:
            # Get cached result
            digest = hashlib.sha256(json.dumps({"q": query, "p": params}, default=str, sort_keys=True).encode("utf-8")).hexdigest()
            cache_key = f"query:{digest}"
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                return cached_result

            # Execute query with connection from pool
            connection = await self.connection_pool.get_connection()
            try:
                # Execute query
                result = {'data': 'query_result'}  # Placeholder

                # Cache result
                await self.cache_manager.set(cache_key, result)

                return result
            finally:
                await self.connection_pool.release_connection(connection)

        except Exception as e:
            logger.error("Error optimizing query: %s", e)
            return {'error': str(e)}

    async def batch_process(
        self,
        items: List[Any],
        batch_type: str
    ) -> Dict[str, Any]:
        """Process items in batches."""
        try:
            for item in items:
                await self.batch_processor.add_to_batch(batch_type, item)

            return {
                'status': 'success',
                'message': f'Added {len(items)} items to {batch_type} batch'
            }

        except Exception as e:
            logger.error("Error batch processing: %s", e)
            return {'error': str(e)}

    async def invalidate_cache(
        self,
        patterns: List[str]
    ) -> Dict[str, Any]:
        """Invalidate cache entries."""
        try:
            for pattern in patterns:
                await self.cache_manager.invalidate(pattern)

            return {
                'status': 'success',
                'message': f'Invalidated cache for {len(patterns)} patterns'
            }

        except Exception as e:
            logger.error("Error invalidating cache: %s", e)
            return {'error': str(e)}

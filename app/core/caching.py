"""
Advanced caching system.
Provides comprehensive caching with compression, encryption, and intelligent eviction.
"""

import asyncio
import hashlib
import logging
import pickle
import threading
import zlib
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from cryptography.fernet import Fernet
from prometheus_client import Counter, Gauge

from app.core.config import get_settings
from app.core.error_handling import ErrorCategory, ErrorSeverity, handle_errors

# Cache metrics
CACHE_HITS = Counter("cache_hits_total", "Cache hits", ["cache_type"])
CACHE_MISSES = Counter("cache_misses_total", "Cache misses", ["cache_type"])
CACHE_SIZE = Gauge("cache_size_bytes", "Cache size in bytes", ["cache_type"])
CACHE_ITEMS = Gauge("cache_items", "Number of cached items", ["cache_type"])
CACHE_OPERATIONS = Counter(
    "cache_operations_total", "Cache operations", ["operation"]
)
EVICTION_COUNT = Counter(
    "cache_evictions_total", "Cache evictions", ["reason"]
)

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class CacheItem:
    """Cache item with metadata."""

    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    size_bytes: int = 0
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    compressed: bool = False
    encrypted: bool = False


class Cache:
    """Advanced caching system."""

    def __init__(self):
        """Initialize cache system."""
        self._lock = threading.Lock()
        self._data: Dict[str, CacheItem] = OrderedDict()
        self._size_bytes: int = 0
        self._fernet = Fernet(
            settings.security.ENCRYPTION_KEY.get_secret_value().encode()
        )
        self._stats: Dict[str, defaultdict] = {
            "hits": defaultdict(int),
            "misses": defaultdict(int),
            "evictions": defaultdict(int),
        }
        self._background_tasks = []

    async def initialize(self):
        """Initialize cache and start background tasks."""
        self._background_tasks = [
            asyncio.create_task(self._cleanup_expired()),
            asyncio.create_task(self._update_metrics()),
            asyncio.create_task(self._optimize_cache()),
        ]

    async def shutdown(self):
        """Shutdown cache and cleanup tasks."""
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._background_tasks.clear()

    @handle_errors(ErrorSeverity.LOW, ErrorCategory.SYSTEM)
    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value from cache."""
        try:
            # Generate cache key
            cache_key = self._generate_key(key)

            with self._lock:
                item = self._data.get(cache_key)
                if not item:
                    # Cache miss
                    self._stats["misses"][key] += 1
                    CACHE_MISSES.labels(cache_type="memory").inc()
                    return default

                # Check expiration
                if item.expires_at and datetime.utcnow() > item.expires_at:
                    # Expired item
                    del self._data[cache_key]
                    self._size_bytes -= item.size_bytes
                    self._stats["misses"][key] += 1
                    CACHE_MISSES.labels(cache_type="memory").inc()
                    return default

                # Update access stats
                item.access_count += 1
                item.last_accessed = datetime.utcnow()

                # Move to end (most recently used)
                self._data.move_to_end(cache_key)

                # Cache hit
                self._stats["hits"][key] += 1
                CACHE_HITS.labels(cache_type="memory").inc()

                # Decrypt if needed
                value = item.value
                if item.encrypted:
                    value = self._decrypt_value(value)

                # Decompress if needed
                if item.compressed:
                    value = self._decompress_value(value)

                return value

        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return default

    @handle_errors(ErrorSeverity.LOW, ErrorCategory.SYSTEM)
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        compress: bool = True,
        encrypt: bool = True,
    ) -> bool:
        """Set a value in cache."""
        try:
            # Generate cache key
            cache_key = self._generate_key(key)

            # Prepare value
            prepared_value = value

            # Compress if needed
            compressed = False
            if compress and self._should_compress(value):
                prepared_value = self._compress_value(prepared_value)
                compressed = True

            # Encrypt if needed
            encrypted = False
            if encrypt:
                prepared_value = self._encrypt_value(prepared_value)
                encrypted = True

            # Calculate size
            size_bytes = self._calculate_size(prepared_value)

            # Check if we need to evict items
            if size_bytes > settings.cache.MAX_SIZE_MB * 1024 * 1024:
                logger.warning(f"Item too large to cache: {size_bytes} bytes")
                return False

            # Create cache item
            item = CacheItem(
                key=cache_key,
                value=prepared_value,
                size_bytes=size_bytes,
                compressed=compressed,
                encrypted=encrypted,
            )

            # Set expiration if TTL provided
            if ttl is not None:
                item.expires_at = datetime.utcnow() + timedelta(seconds=ttl)

            with self._lock:
                # Remove old item if exists
                if cache_key in self._data:
                    old_item = self._data[cache_key]
                    self._size_bytes -= old_item.size_bytes

                # Check if we need to evict items
                while (
                    self._size_bytes + size_bytes
                    > settings.cache.MAX_SIZE_MB * 1024 * 1024
                ):
                    if not self._evict_item():
                        logger.error("Failed to evict cache item")
                        return False

                # Store new item
                self._data[cache_key] = item
                self._size_bytes += size_bytes

                CACHE_OPERATIONS.labels(operation="set").inc()
                return True

        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")
            return False

    @handle_errors(ErrorSeverity.LOW, ErrorCategory.SYSTEM)
    async def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        try:
            # Generate cache key
            cache_key = self._generate_key(key)

            with self._lock:
                if cache_key in self._data:
                    item = self._data[cache_key]
                    self._size_bytes -= item.size_bytes
                    del self._data[cache_key]
                    CACHE_OPERATIONS.labels(operation="delete").inc()
                    return True
                return False

        except Exception as e:
            logger.error(f"Cache delete error: {str(e)}")
            return False

    @handle_errors(ErrorSeverity.LOW, ErrorCategory.SYSTEM)
    async def clear(self) -> bool:
        """Clear all items from cache."""
        try:
            with self._lock:
                self._data.clear()
                self._size_bytes = 0
                CACHE_OPERATIONS.labels(operation="clear").inc()
                return True

        except Exception as e:
            logger.error(f"Cache clear error: {str(e)}")
            return False

    def _generate_key(self, key: str) -> str:
        """Generate a cache key."""
        return hashlib.sha256(str(key).encode()).hexdigest()

    def _calculate_size(self, value: Any) -> int:
        """Calculate size of a value in bytes."""
        return len(pickle.dumps(value))

    def _should_compress(self, value: Any) -> bool:
        """Determine if a value should be compressed."""
        try:
            # Only compress strings and bytes over compression threshold
            if isinstance(value, (str, bytes)):
                size = len(
                    value if isinstance(value, bytes) else value.encode()
                )
                return size >= settings.cache.COMPRESSION_MIN_SIZE
            return False
        except Exception:
            return False

    def _compress_value(self, value: Any) -> bytes:
        """Compress a value."""
        try:
            if isinstance(value, str):
                value = value.encode()
            return zlib.compress(value)
        except Exception as e:
            logger.error(f"Compression error: {str(e)}")
            return value

    def _decompress_value(self, value: bytes) -> Any:
        """Decompress a value."""
        try:
            return zlib.decompress(value)
        except Exception as e:
            logger.error(f"Decompression error: {str(e)}")
            return value

    def _encrypt_value(self, value: Any) -> bytes:
        """Encrypt a value."""
        try:
            if not isinstance(value, bytes):
                value = pickle.dumps(value)
            return self._fernet.encrypt(value)
        except Exception as e:
            logger.error(f"Encryption error: {str(e)}")
            return value

    def _decrypt_value(self, value: bytes) -> Any:
        """Decrypt a value."""
        try:
            decrypted = self._fernet.decrypt(value)
            return pickle.loads(decrypted)
        except Exception as e:
            logger.error(f"Decryption error: {str(e)}")
            return value

    def _evict_item(self) -> bool:
        """Evict a cache item based on policy."""
        try:
            if not self._data:
                return False

            with self._lock:
                # Get item to evict
                key, item = self._get_item_to_evict()
                if not key:
                    return False

                # Remove item
                self._size_bytes -= item.size_bytes
                del self._data[key]

                # Update stats
                self._stats["evictions"][item.key] += 1
                EVICTION_COUNT.labels(reason="size").inc()

                return True

        except Exception as e:
            logger.error(f"Cache eviction error: {str(e)}")
            return False

    def _get_item_to_evict(self) -> Tuple[Optional[str], Optional[CacheItem]]:
        """Get the next item to evict."""
        try:
            # First, try to find expired items
            current_time = datetime.utcnow()
            expired = [
                (k, v)
                for k, v in self._data.items()
                if v.expires_at and current_time > v.expires_at
            ]
            if expired:
                return expired[0]

            # Then, try least recently used
            if self._data:
                key = next(iter(self._data))
                return key, self._data[key]

            return None, None

        except Exception as e:
            logger.error(f"Get item to evict error: {str(e)}")
            return None, None

    async def _cleanup_expired(self):
        """Clean up expired cache items."""
        while True:
            try:
                current_time = datetime.utcnow()

                with self._lock:
                    # Find expired items
                    expired = [
                        key
                        for key, item in self._data.items()
                        if item.expires_at and current_time > item.expires_at
                    ]

                    # Remove expired items
                    for key in expired:
                        item = self._data[key]
                        self._size_bytes -= item.size_bytes
                        del self._data[key]
                        EVICTION_COUNT.labels(reason="expired").inc()

                await asyncio.sleep(60)  # Run every minute

            except Exception as e:
                logger.error(f"Cache cleanup error: {str(e)}")
                await asyncio.sleep(60)

    async def _update_metrics(self):
        """Update cache metrics."""
        while True:
            try:
                with self._lock:
                    # Update size metrics
                    CACHE_SIZE.labels(cache_type="memory").set(
                        self._size_bytes
                    )
                    CACHE_ITEMS.labels(cache_type="memory").set(
                        len(self._data)
                    )

                await asyncio.sleep(15)  # Update every 15 seconds

            except Exception as e:
                logger.error(f"Cache metrics update error: {str(e)}")
                await asyncio.sleep(60)

    async def _optimize_cache(self):
        """Optimize cache based on usage patterns."""
        while True:
            try:
                with self._lock:
                    if not self._data:
                        continue

                    # Calculate average access frequency
                    total_accesses = sum(
                        item.access_count for item in self._data.values()
                    )
                    avg_accesses = total_accesses / len(self._data)

                    # Find items with low access counts
                    low_access_items = [
                        key
                        for key, item in self._data.items()
                        if (
                            item.access_count < avg_accesses * 0.2
                            and (  # 20% of average
                                datetime.utcnow()
                                -
                                # 1 hour
                                item.last_accessed
                            ).total_seconds()
                            > 3600
                        )
                    ]

                    # Remove low access items
                    for key in low_access_items:
                        item = self._data[key]
                        self._size_bytes -= item.size_bytes
                        del self._data[key]
                        EVICTION_COUNT.labels(reason="optimization").inc()

                await asyncio.sleep(300)  # Run every 5 minutes

            except Exception as e:
                logger.error(f"Cache optimization error: {str(e)}")
                await asyncio.sleep(60)

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            with self._lock:
                return {
                    "size_bytes": self._size_bytes,
                    "item_count": len(self._data),
                    "hits": dict(self._stats["hits"]),
                    "misses": dict(self._stats["misses"]),
                    "evictions": dict(self._stats["evictions"]),
                    "memory_usage": self._size_bytes
                    / (settings.cache.MAX_SIZE_MB * 1024 * 1024),
                }

        except Exception as e:
            logger.error(f"Get cache stats error: {str(e)}")
            return {}


# Global cache instance
cache = Cache()

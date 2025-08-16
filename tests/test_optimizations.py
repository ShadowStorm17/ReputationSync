"""
Tests for API optimizations.
Tests the performance, security, and reliability enhancements.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, AsyncMock
from app.core.optimizations import (
    HierarchicalCache,
    CircuitBreaker,
    RateLimiter,
    cache_warmer,
    QueryOptimizer,
    PerformanceMonitor
)

@pytest.fixture
async def hierarchical_cache():
    """Fixture for hierarchical cache."""
    return HierarchicalCache()

@pytest.fixture
async def circuit_breaker():
    """Fixture for circuit breaker."""
    return CircuitBreaker(failure_threshold=3, reset_timeout=30)

@pytest.fixture
async def rate_limiter():
    """Fixture for rate limiter."""
    return RateLimiter(requests=10, window=60)

class TestHierarchicalCache:
    """Test cases for hierarchical cache."""
    
    @pytest.mark.asyncio
    async def test_cache_get_l1(self, hierarchical_cache):
        """Test L1 cache retrieval."""
        # Set value in L1 cache
        hierarchical_cache.l1_cache["test_key"] = "test_value"
        
        # Get value
        result = await hierarchical_cache.get("test_key")
        
        assert result == "test_value"
    
    @pytest.mark.asyncio
    async def test_cache_get_l2(self, hierarchical_cache):
        """Test L2 cache retrieval."""
        # Mock L2 cache
        hierarchical_cache.l2_cache.get = AsyncMock(return_value="test_value")
        
        # Get value
        result = await hierarchical_cache.get("test_key")
        
        assert result == "test_value"
        assert "test_key" in hierarchical_cache.l1_cache
    
    @pytest.mark.asyncio
    async def test_cache_set(self, hierarchical_cache):
        """Test cache set operation."""
        # Mock L2 cache
        hierarchical_cache.l2_cache.set = AsyncMock()
        
        # Set value
        await hierarchical_cache.set("test_key", "test_value", 60)
        
        assert hierarchical_cache.l1_cache["test_key"] == "test_value"
        hierarchical_cache.l2_cache.set.assert_called_once_with(
            "test_key",
            "test_value",
            60
        )

class TestCircuitBreaker:
    """Test cases for circuit breaker."""
    
    @pytest.mark.asyncio
    async def test_successful_call(self, circuit_breaker):
        """Test successful function call."""
        @circuit_breaker
        async def test_func():
            return "success"
        
        result = await test_func()
        assert result == "success"
        assert circuit_breaker.state == "closed"
    
    @pytest.mark.asyncio
    async def test_failed_calls(self, circuit_breaker):
        """Test circuit breaker opening on failures."""
        @circuit_breaker
        async def test_func():
            raise Exception("test error")
        
        # Trigger failures
        for _ in range(3):
            with pytest.raises(Exception):
                await test_func()
        
        assert circuit_breaker.state == "open"
    
    @pytest.mark.asyncio
    async def test_reset_timeout(self, circuit_breaker):
        """Test circuit breaker reset timeout."""
        circuit_breaker.state = "open"
        circuit_breaker.last_failure_time = datetime.now(timezone.utc) - timedelta(seconds=31)
        
        @circuit_breaker
        async def test_func():
            return "success"
        
        result = await test_func()
        assert result == "success"
        assert circuit_breaker.state == "closed"

class TestRateLimiter:
    """Test cases for rate limiter."""
    
    @pytest.mark.asyncio
    async def test_allowed_request(self, rate_limiter):
        """Test allowed request."""
        # Mock Redis
        rate_limiter.redis.get = AsyncMock(return_value=None)
        rate_limiter.redis.setex = AsyncMock()
        
        result = await rate_limiter.is_allowed("test_key")
        
        assert result is True
        rate_limiter.redis.setex.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, rate_limiter):
        """Test rate limit exceeded."""
        # Mock Redis
        rate_limiter.redis.get = AsyncMock(return_value="10")
        
        result = await rate_limiter.is_allowed("test_key")
        
        assert result is False

class TestQueryOptimizer:
    """Test cases for query optimizer."""
    
    @pytest.mark.asyncio
    async def test_optimize_query(self):
        """Test query optimization."""
        query = "SELECT * FROM users"
        result = await QueryOptimizer.optimize_query(query)
        
        assert isinstance(result, str)
    
    def test_materialized_view_check(self):
        """Test materialized view check."""
        query = "SELECT COUNT(*) FROM large_table"
        result = QueryOptimizer.should_use_materialized_view(query)
        
        assert isinstance(result, bool)

class TestPerformanceMonitor:
    """Test cases for performance monitor."""
    
    @pytest.mark.asyncio
    async def test_record_slow_query(self):
        """Test slow query recording."""
        monitor = PerformanceMonitor()
        
        await monitor.record_query(
            "SELECT * FROM users",
            2.5  # 2.5 seconds
        )
        
        assert len(monitor.slow_queries) == 1
        assert monitor.slow_queries[0]["duration"] == 2.5
    
    @pytest.mark.asyncio
    async def test_record_response_time(self):
        """Test response time recording."""
        monitor = PerformanceMonitor()
        
        await monitor.record_response_time(
            "/api/users",
            0.5  # 500ms
        )
        
        assert len(monitor.response_times) == 1
        assert monitor.response_times[0]["duration"] == 0.5 
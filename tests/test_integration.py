import pytest
import redis.asyncio as redis
import time
import hashlib
import secrets
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app
from app.core.config import get_settings
from app.core.cache import RedisCache
from app.core.security import APIKeyManager, RequestValidator
from app.services.api_key_manager import APIKeyManager

settings = get_settings()

@pytest.fixture
async def redis_client():
    """Create a Redis client for testing."""
    redis = await redis.from_url(
        "redis://localhost:6379",
        encoding="utf-8",
        decode_responses=True
    )
    yield redis
    await redis.flushdb()  # Clean up after tests
    await redis.close()

@pytest.fixture
def test_client():
    """Create a test client."""
    return TestClient(app)

@pytest.fixture
async def api_key_manager():
    """Create an API key manager instance."""
    return APIKeyManager()

@pytest.fixture
async def test_api_key(api_key_manager):
    """Create a test API key."""
    key_data = await api_key_manager.create_api_key("test_client")
    return key_data["api_key"]

class TestCacheIntegration:
    async def test_cache_operations(self, redis_client):
        """Test Redis cache operations."""
        cache = RedisCache()
        test_key = "test:key"
        test_data = {"test": "data"}
        
        # Test set
        success = await cache.set(test_key, test_data)
        assert success is True
        
        # Test get
        cached_data = await cache.get(test_key)
        assert cached_data == test_data
        
        # Test delete
        success = await cache.delete(test_key)
        assert success is True
        
        # Verify deletion
        cached_data = await cache.get(test_key)
        assert cached_data is None

class TestAPIKeyIntegration:
    async def test_api_key_lifecycle(self, api_key_manager, redis_client):
        """Test API key creation, validation, and revocation."""
        # Create key
        key_data = await api_key_manager.create_api_key("test_client")
        api_key = key_data["api_key"]
        
        # Validate key
        is_valid = await api_key_manager.validate_api_key(api_key)
        assert is_valid is True
        
        # Revoke key
        revoked = await api_key_manager.revoke_api_key(api_key)
        assert revoked is True
        
        # Verify revocation
        is_valid = await api_key_manager.validate_api_key(api_key)
        assert is_valid is False

class TestRequestValidation:
    def test_request_signature(self, test_client, test_api_key):
        """Test request signature validation."""
        validator = RequestValidator()
        timestamp = str(int(time.time()))
        nonce = secrets.token_hex(16)
        test_body = '{"test": "data"}'
        
        # Generate signature
        message = f"{timestamp}{nonce}{test_api_key}{test_body}"
        signature = hashlib.sha256(message.encode()).hexdigest()
        
        # Make request with valid signature
        headers = {
            "X-API-Key": test_api_key,
            "X-Timestamp": timestamp,
            "X-Nonce": nonce,
            "X-Signature": signature,
            "Content-Type": "application/json"
        }
        
        response = test_client.post(
            "/api/v1/test",
            headers=headers,
            json={"test": "data"}
        )
        
        assert response.status_code != 401

class TestInstagramIntegration:
    @pytest.mark.asyncio
    async def test_instagram_api_with_cache(self, test_client, test_api_key, redis_client):
        """Test Instagram API with caching."""
        username = "test_user"
        headers = {"X-API-Key": test_api_key}
        
        # Mock Instagram API response
        mock_user_data = {
            "username": username,
            "followers_count": 1000,
            "follows_count": 500,
            "media_count": 100,
            "is_private": False
        }
        
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"data": [{"id": "123"}]}
            
            # First request - should hit Instagram API
            response = test_client.get(
                f"/api/v1/platforms/instagram/users/{username}",
                headers=headers
            )
            
            assert response.status_code == 200
            assert mock_get.call_count > 0
            
            # Second request - should hit cache
            response = test_client.get(
                f"/api/v1/platforms/instagram/users/{username}",
                headers=headers
            )
            
            assert response.status_code == 200
            # Verify call count hasn't increased (cache hit)
            assert mock_get.call_count == 1

class TestRateLimiting:
    def test_rate_limiting(self, test_client, test_api_key):
        """Test rate limiting functionality."""
        headers = {"X-API-Key": test_api_key}
        
        # Make requests up to limit
        for _ in range(settings.RATE_LIMIT_CALLS):
            response = test_client.get(
                "/api/v1/platforms/instagram/users/test_user",
                headers=headers
            )
            assert response.status_code != 429
        
        # Next request should be rate limited
        response = test_client.get(
            "/api/v1/platforms/instagram/users/test_user",
            headers=headers
        )
        assert response.status_code == 429

@pytest.mark.asyncio
async def test_api_versioning():
    """Test API versioning support."""
    client = TestClient(app)
    
    # Test v1 (deprecated) endpoint
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    assert "deprecated" in response.headers.get("Warning", "")
    
    # Test v2 endpoint
    response = client.get("/api/v2/status")
    assert response.status_code == 200
    assert "deprecated" not in response.headers.get("Warning", "") 
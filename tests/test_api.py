import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta, timezone
import json

def test_health_check(test_client: TestClient):
    """Test the health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data

@pytest.mark.asyncio
async def test_instagram_user_info(
    test_client: TestClient,
    test_api_key: dict,
    security_service: "SecurityService"
):
    """Test the Instagram user info endpoint."""
    username = "test_user"
    
    # Generate valid request signature
    headers = security_service.generate_signature(
        test_api_key["key"],
        test_api_key["key"]
    )
    
    response = test_client.get(
        f"/api/v1/platforms/instagram/users/{username}",
        headers=headers
    )
    
    assert response.status_code in [200, 404]  # Either success or user not found
    if response.status_code == 200:
        data = response.json()
        assert data["username"] == username
        assert "follower_count" in data
        assert "following_count" in data
        assert "is_private" in data
        assert "post_count" in data

@pytest.mark.asyncio
async def test_rate_limiting(
    test_client: TestClient,
    test_api_key: dict,
    security_service: "SecurityService"
):
    """Test rate limiting functionality."""
    headers = security_service.generate_signature(
        test_api_key["key"],
        test_api_key["key"]
    )
    
    # Make requests until rate limit is hit
    # API endpoints have a limit of 1000 requests per minute
    responses = []
    for _ in range(1010):  # API tier limit is 1000/minute
        response = test_client.get(
            "/api/v1/status",
            headers=headers
        )
        responses.append(response)
        if response.status_code == 429:
            break
    
    # Verify rate limit was hit
    assert any(r.status_code == 429 for r in responses)
    
    # Verify rate limit headers
    success_response = next(r for r in responses if r.status_code == 200)
    assert "X-RateLimit-Limit" in success_response.headers
    assert "X-RateLimit-Remaining" in success_response.headers
    assert "X-RateLimit-Reset" in success_response.headers

@pytest.mark.asyncio
async def test_api_key_management(
    test_client: TestClient,
    test_api_key: dict,
    security_service: "SecurityService"
):
    """Test API key management endpoints."""
    headers = security_service.generate_signature(
        test_api_key["key"],
        test_api_key["key"]
    )
    
    # Create new API key
    response = test_client.post(
        "/create_api_key",
        headers=headers,
        json={"name": "test_key_2"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "key" in data
    new_key = data["key"]
    
    # Revoke API key
    response = test_client.post(
        f"/api/keys/{new_key}/revoke",
        headers=headers
    )
    assert response.status_code == 200
    
    # Verify revoked key doesn't work
    new_headers = security_service.generate_signature(
        new_key,
        new_key
    )
    response = test_client.get(
        "/api/v1/status",
        headers=new_headers
    )
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_request_signing(
    test_client: TestClient,
    test_api_key: dict,
    security_service: "SecurityService"
):
    """Test request signing validation."""
    # Test with valid signature
    headers = security_service.generate_signature(
        test_api_key["key"],
        test_api_key["key"]
    )
    response = test_client.get(
        "/api/v1/status",
        headers=headers
    )
    assert response.status_code == 200
    
    # Test with invalid signature
    headers["X-Signature"] = "invalid_signature"
    response = test_client.get(
        "/api/v1/status",
        headers=headers
    )
    assert response.status_code == 401
    
    # Test with expired timestamp
    old_time = int(datetime.now(timezone.utc).timestamp()) - 600  # 10 minutes ago
    headers = security_service.generate_signature(
        test_api_key["key"],
        test_api_key["key"]
    )
    headers["X-Timestamp"] = str(old_time)
    response = test_client.get(
        "/api/v1/status",
        headers=headers
    )
    assert response.status_code == 401
    
    # Test with reused nonce
    headers = security_service.generate_signature(
        test_api_key["key"],
        test_api_key["key"]
    )
    response1 = test_client.get(
        "/api/v1/status",
        headers=headers
    )
    response2 = test_client.get(
        "/api/v1/status",
        headers=headers
    )
    assert response1.status_code == 200
    assert response2.status_code == 401  # Second request should fail 
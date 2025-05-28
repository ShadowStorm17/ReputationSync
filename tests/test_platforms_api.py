import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app
from app.core.config import get_settings

settings = get_settings()
client = TestClient(app)

# Mock API key for testing
TEST_API_KEY = "test-api-key"

# Override settings for testing
@pytest.fixture(autouse=True)
def mock_settings():
    with patch("app.core.security.settings") as mock_settings:
        mock_settings.SECRET_KEY = TEST_API_KEY
        yield mock_settings

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_get_instagram_user_success():
    """Test successful Instagram user info retrieval."""
    headers = {"X-API-Key": TEST_API_KEY}
    
    with patch('app.services.instagram_service.InstagramService.get_user_info') as mock_get_user:
        mock_get_user.return_value = {
            "username": "testuser",
            "follower_count": 1000,
            "following_count": 500,
            "is_private": False,
            "post_count": 100
        }
        
        response = client.get("/api/v1/platforms/instagram/users/testuser", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "testuser"
        assert data["follower_count"] == 1000
        assert data["following_count"] == 500
        assert data["is_private"] is False
        assert data["post_count"] == 100

def test_get_instagram_user_no_api_key():
    """Test request without API key."""
    response = client.get("/api/v1/platforms/instagram/users/testuser")
    assert response.status_code == 401
    assert "Invalid API Key" in response.json()["detail"]

def test_get_instagram_user_invalid_api_key():
    """Test request with invalid API key."""
    headers = {"X-API-Key": "invalid-key"}
    response = client.get("/api/v1/platforms/instagram/users/testuser", headers=headers)
    assert response.status_code == 401
    assert "Invalid API Key" in response.json()["detail"]

def test_get_instagram_user_service_error():
    """Test handling of Instagram service errors."""
    headers = {"X-API-Key": TEST_API_KEY}
    
    with patch('app.services.instagram_service.InstagramService.get_user_info') as mock_get_user:
        mock_get_user.side_effect = Exception("Instagram API error")
        response = client.get("/api/v1/platforms/instagram/users/testuser", headers=headers)
        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]
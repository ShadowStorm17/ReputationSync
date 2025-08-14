import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app
from app.core.config import get_settings
import os

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

print('INSTAGRAM_ACCESS_TOKEN:', os.getenv('INSTAGRAM_ACCESS_TOKEN'))

REAL_INSTAGRAM_USERNAME = "me"

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_get_instagram_user_success():
    headers = {
        "X-API-Key": TEST_API_KEY,
        "X-Timestamp": "1234567890",
        "X-Nonce": "testnonce",
        "X-Signature": "dummysignature"
        }
    response = client.get(f"/api/v1/platforms/instagram/users/{REAL_INSTAGRAM_USERNAME}", headers=headers)
        assert response.status_code == 200
        data = response.json()
    assert data["username"]
    assert "id" in data
    assert "account_type" in data
    assert "media_count" in data


def test_get_instagram_user_no_api_key():
    response = client.get("/api/v1/platforms/instagram/users/testuser")
    assert response.status_code == 401
    assert "Invalid API Key" in response.json()["detail"]


def test_get_instagram_user_invalid_api_key():
    headers = {"X-API-Key": "invalid-key"}
    response = client.get("/api/v1/platforms/instagram/users/testuser", headers=headers)
    assert response.status_code == 401
    assert "Invalid API Key" in response.json()["detail"]


def test_get_instagram_user_service_error():
    headers = {
        "X-API-Key": TEST_API_KEY,
        "X-Timestamp": "1234567890",
        "X-Nonce": "testnonce",
        "X-Signature": "dummysignature"
    }
    response = client.get(f"/api/v1/platforms/instagram/users/{REAL_INSTAGRAM_USERNAME}", headers=headers)
    assert response.status_code in (200, 500, 404, 400)
    if response.status_code == 500:
        assert "temporarily unavailable" in response.json()["detail"]
    if response.status_code == 404:
        assert "User not found" in response.json()["detail"]
    if response.status_code == 400:
        assert "Only the authenticated user" in response.json()["detail"] or "Invalid or expired access token" in response.json()["detail"]
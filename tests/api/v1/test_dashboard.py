import pytest
from fastapi import status
from unittest.mock import patch, MagicMock, AsyncMock

@pytest.mark.asyncio
async def test_dashboard_unauthorized(client):
    """Test dashboard access without authentication."""
    response = client.get("/api/v1/dashboard/")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

@pytest.mark.asyncio
async def test_dashboard_authorized(authorized_client):
    """Test dashboard access with authentication."""
    response = authorized_client.get("/api/v1/dashboard/")
    assert response.status_code == status.HTTP_200_OK
    # Check for expected keys in the metrics summary
    data = response.json()
    assert "system" in data
    assert "reputation" in data

@pytest.mark.asyncio
async def test_dashboard_refresh_unauthorized(client):
    """Test dashboard refresh without authentication."""
    response = client.get("/api/v1/dashboard/refresh")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

@pytest.mark.asyncio
async def test_dashboard_refresh_authorized(authorized_client):
    """Test dashboard refresh with authentication."""
    response = authorized_client.get("/api/v1/dashboard/refresh")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "system" in data
    assert "reputation" in data 
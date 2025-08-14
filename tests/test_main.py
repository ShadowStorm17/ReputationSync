import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "Reputation Management API"
    assert response.json()["status"] == "operational"

def test_platforms():
    """Test platforms endpoint."""
    response = client.get("/platforms")
    assert response.status_code == 401  # Should require authentication

def test_sentiment_analysis():
    """Test sentiment analysis endpoint."""
    response = client.post("/api/v1/analyze/sentiment", json={"text": "Great service!"})
    assert response.status_code == 401  # Should require authentication

def test_response_generation():
    """Test response generation endpoint."""
    data = {
        "message": "I love your product!",
        "context": {
            "user_name": "TestUser",
            "reference_id": "123"
        }
    }
    response = client.post("/api/v1/response/generate", json=data)
    assert response.status_code == 401  # Should require authentication

def test_monitor_start():
    """Test monitoring start endpoint."""
    config = {
        "platforms": ["twitter"],
        "keywords": ["test brand"],
        "interval": 300
    }
    response = client.post("/api/v1/monitoring/start", json=config)
    assert response.status_code == 401  # Should require authentication

def test_monitor_stop():
    """Test monitoring stop endpoint."""
    response = client.post("/api/v1/monitoring/stop", json={})
    assert response.status_code == 401  # Should require authentication

def test_crisis_check():
    """Test crisis check endpoint."""
    metrics = {
        "mention_rate_change": 2.5,
        "sentiment_distribution": {
            "positive": 10,
            "neutral": 15,
            "negative": 25
        },
        "engagement_change": -0.3
    }
    response = client.post("/api/v1/crisis/check", json=metrics)
    assert response.status_code == 401  # Should require authentication

@pytest.mark.asyncio
async def test_websocket():
    """Test WebSocket connection."""
    with client.websocket_connect("/ws/test_client") as websocket:
        # Send ping
        websocket.send_json({"type": "ping"})
        # Receive pong
        data = websocket.receive_json()
        assert data["type"] == "pong" 
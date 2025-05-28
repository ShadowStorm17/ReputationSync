"""
Load testing script for Instagram Stats API.
Run with: locust -f locustfile.py
"""

import time
import json
import hmac
import hashlib
from typing import Dict, Optional
from locust import HttpUser, task, between, events
from datetime import datetime, timedelta

class InstagramStatsUser(HttpUser):
    """Simulated user for load testing."""
    
    # Wait between 1 and 5 seconds between tasks
    wait_time = between(1, 5)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_key = None
        self.client_id = None
        self.client_secret = None
    
    def on_start(self):
        """Initialize user with API key."""
        # Get API key
        response = self.get_api_key()
        if response.status_code == 200:
            data = response.json()
            self.api_key = data["access_token"]
    
    def get_api_key(self) -> Dict:
        """Get API key for authentication."""
        self.client_id = "test_client"
        self.client_secret = "test_secret"
        
        with self.client.post(
            "/api/v1/auth/token",
            json={
                "client_id": self.client_id,
                "client_secret": self.client_secret
            },
            name="/auth/token",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to get API key: {response.text}")
            return response
    
    def sign_request(self, method: str, path: str, body: Optional[str] = None) -> Dict[str, str]:
        """Generate request signature."""
        timestamp = str(int(time.time()))
        message = f"{method}\n{path}\n{timestamp}\n{body or ''}"
        signature = hmac.new(
            key=self.api_key.encode(),
            msg=message.encode(),
            digestmod=hashlib.sha256
        ).hexdigest()
        
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-Request-Timestamp": timestamp,
            "X-Request-Signature": signature
        }
    
    @task(1)
    def get_profile_stats(self):
        """Test profile statistics endpoint."""
        headers = self.sign_request("GET", "/api/v1/stats/profile")
        
        with self.client.get(
            "/api/v1/stats/profile",
            headers=headers,
            name="/stats/profile",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to get profile stats: {response.text}")
    
    @task(2)
    def get_post_stats(self):
        """Test post statistics endpoint."""
        # Get last 7 days of posts
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        headers = self.sign_request(
            "GET",
            f"/api/v1/stats/posts?start_date={start_date}&end_date={end_date}"
        )
        
        with self.client.get(
            f"/api/v1/stats/posts?start_date={start_date}&end_date={end_date}",
            headers=headers,
            name="/stats/posts",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to get post stats: {response.text}")
    
    @task(1)
    def get_engagement_stats(self):
        """Test engagement statistics endpoint."""
        headers = self.sign_request(
            "GET",
            "/api/v1/stats/engagement?period=week"
        )
        
        with self.client.get(
            "/api/v1/stats/engagement?period=week",
            headers=headers,
            name="/stats/engagement",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to get engagement stats: {response.text}")
    
    @task(1)
    def get_audience_stats(self):
        """Test audience statistics endpoint."""
        headers = self.sign_request("GET", "/api/v1/stats/audience")
        
        with self.client.get(
            "/api/v1/stats/audience",
            headers=headers,
            name="/stats/audience",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to get audience stats: {response.text}")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    print("Starting load test...")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops."""
    print("Load test completed.")

class WebsiteUser(HttpUser):
    """
    Simulated website user for testing rate limiting and error handling.
    This user doesn't authenticate properly to test error cases.
    """
    
    wait_time = between(1, 3)
    
    @task(1)
    def test_rate_limiting(self):
        """Test rate limiting by making rapid requests."""
        with self.client.get(
            "/api/v1/stats/profile",
            name="/stats/profile (no auth)",
            catch_response=True
        ) as response:
            if response.status_code == 401:
                response.success()
            else:
                response.failure("Expected 401 unauthorized")
    
    @task(1)
    def test_invalid_signature(self):
        """Test invalid request signing."""
        headers = {
            "Authorization": "Bearer invalid_key",
            "X-Request-Timestamp": str(int(time.time())),
            "X-Request-Signature": "invalid_signature"
        }
        
        with self.client.get(
            "/api/v1/stats/profile",
            headers=headers,
            name="/stats/profile (invalid auth)",
            catch_response=True
        ) as response:
            if response.status_code == 401:
                response.success()
            else:
                response.failure("Expected 401 unauthorized")
    
    @task(1)
    def test_invalid_endpoint(self):
        """Test non-existent endpoint."""
        with self.client.get(
            "/api/v1/invalid",
            name="/invalid",
            catch_response=True
        ) as response:
            if response.status_code == 404:
                response.success()
            else:
                response.failure("Expected 404 not found") 
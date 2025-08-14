from locust import HttpUser, task, between
from app.core.security import SecurityService
import random

class APIUser(HttpUser):
    wait_time = between(1, 2)
    
    def on_start(self):
        """Setup before tests."""
        self.security = SecurityService()
        self.api_key = "test_key_123"  # Replace with actual test key
        self.usernames = [
            "test_user1",
            "test_user2",
            "test_user3",
            "test_user4",
            "test_user5"
        ]
    
    def _get_headers(self, body: str = ""):
        """Get signed request headers."""
        return self.security.generate_signature(
            self.api_key,
            self.api_key,
            body
        )
    
    @task(1)
    def health_check(self):
        """Test health check endpoint."""
        self.client.get("/health")
    
    @task(3)
    def get_user_info(self):
        """Test Instagram user info endpoint."""
        username = random.choice(self.usernames)
        headers = self._get_headers()
        self.client.get(
            f"/api/v1/platforms/instagram/users/{username}",
            headers=headers
        )
    
    @task(2)
    def api_status(self):
        """Test API status endpoint."""
        headers = self._get_headers()
        self.client.get(
            "/api/v1/status",
            headers=headers
        )

class BulkAPIUser(HttpUser):
    """User class for testing bulk operations."""
    wait_time = between(5, 10)
    
    def on_start(self):
        """Setup before tests."""
        self.security = SecurityService()
        self.api_key = "test_key_123"  # Replace with actual test key
        self.usernames = [f"bulk_user_{i}" for i in range(100)]
    
    def _get_headers(self, body: str = ""):
        """Get signed request headers."""
        return self.security.generate_signature(
            self.api_key,
            self.api_key,
            body
        )
    
    @task
    def bulk_user_info(self):
        """Test bulk user info endpoint."""
        # Select 10 random usernames
        selected_users = random.sample(self.usernames, 10)
        headers = self._get_headers(body=str(selected_users))
        
        self.client.post(
            "/api/v1/platforms/instagram/users/bulk",
            headers=headers,
            json={"usernames": selected_users}
        ) 
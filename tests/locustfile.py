from locust import HttpUser, task, between
import random

class InstagramAPIUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Setup test API key."""
        self.api_key = "test-api-key"  # Replace with test API key
        self.headers = {"X-API-Key": self.api_key}
        self.test_usernames = [
            "test_user1",
            "test_user2",
            "instagram",
            "cristiano",
            "leomessi"
        ]
    
    @task(3)
    def get_user_info(self):
        """Test the main user info endpoint."""
        username = random.choice(self.test_usernames)
        self.client.get(
            f"/api/v1/platforms/instagram/users/{username}",
            headers=self.headers
        )
    
    @task(1)
    def health_check(self):
        """Test the health check endpoint."""
        self.client.get("/health")
    
    @task(1)
    def api_status(self):
        """Test the API status endpoint."""
        self.client.get("/api/v1/status", headers=self.headers) 
import requests
import logging
import time
from datetime import datetime
from typing import Optional
import os

logger = logging.getLogger(__name__)

class APIClient:
    """Client for interacting with the ReputationSync API."""

    def __init__(self, api_key: str, base_url: str = "https://api.reputationsync.com", timeout: float | int | None = None):
        """
        Initialize the API client.
        
        Args:
            api_key (str): The API key for authentication
            base_url (str): The base URL of the API
            timeout (float|int|None): Timeout (seconds) for HTTP requests; defaults to 30s
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
        # Default timeout for all requests
        env_timeout = os.getenv('API_CLIENT_TIMEOUT_SECONDS')
        try:
            self.timeout: float | None = float(env_timeout) if env_timeout is not None else 30.0
        except ValueError:
            self.timeout = 30.0
        if timeout is not None:
            self.timeout = float(timeout)

    def record_usage(self, endpoint: str, user_id: int, response_time: Optional[int] = None, success: bool = True):
        """
        Record API usage statistics.
        
        Args:
            endpoint (str): The API endpoint that was called
            user_id (int): The ID of the user making the request
            response_time (int, optional): Response time in milliseconds. If not provided, will be calculated from start_time
            success (bool): Whether the request was successful
        """
        try:
            data = {
                'endpoint': endpoint,
                'user_id': user_id,
                'response_time': response_time or 0,
                'success': success
            }
            
            response = self.session.post(
                f'{self.base_url}/api/record-usage',
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error("Failed to record API usage: %s", e)
            return None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        if self.start_time is not None:
            self.last_response_time = int((end_time - self.start_time) * 1000)  # Convert to milliseconds
        else:
            self.last_response_time = 0
        # Ensure the underlying session is closed to prevent resource leaks
        try:
            self.session.close()
        except Exception as e:
            logger.debug("Error closing APIClient session: %s", e)
        return False  # Don't suppress exceptions

def example_usage():
    # Example usage of the API client
    api_key = os.getenv('API_KEY')
    if not api_key:
        logger.warning("API_KEY not set in environment; skipping example_usage.")
        return
    client = APIClient(api_key)

    # Example 1: Basic usage
    client.record_usage(
        endpoint='/api/data',
        user_id=123,
        response_time=150,
        success=True
    )

    # Example 2: Using context manager to automatically track response time
    with APIClient(api_key) as client:
        # Your API call here
        time.sleep(1)  # Simulate API call
        client.record_usage(
            endpoint='/api/users',
            user_id=456,
            response_time=client.last_response_time,
            success=True
        )

if __name__ == '__main__':
    example_usage() 
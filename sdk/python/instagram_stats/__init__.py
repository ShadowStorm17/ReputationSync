"""
Instagram Stats API Python SDK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Python SDK for the Instagram Stats API.

Basic usage:

    >>> from instagram_stats import InstagramStatsClient
    >>> client = InstagramStatsClient('your_client_id', 'your_client_secret')
    >>> profile = client.get_profile_stats()
    >>> print(profile.followers_count)
    10000
"""

import time
import hmac
import hashlib
import requests
from typing import Dict, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime, date

@dataclass
class ProfileStats:
    followers_count: int
    following_count: int
    media_count: int
    engagement_rate: float
    profile_views: int
    website_clicks: int
    email_clicks: int
    updated_at: datetime

@dataclass
class PostStats:
    id: str
    type: str
    caption: str
    likes: int
    comments: int
    saves: int
    shares: int
    reach: int
    impressions: int
    engagement_rate: float
    posted_at: datetime

@dataclass
class EngagementStats:
    total_engagement: int
    engagement_rate: float
    breakdown: Dict[str, int]
    trend: List[Dict[str, Union[str, float, int]]]

@dataclass
class AudienceStats:
    total_followers: int
    growth_rate: float
    demographics: Dict[str, Dict[str, float]]
    online_times: Dict[str, Union[str, int, List[Dict[str, float]]]]

class InstagramStatsError(Exception):
    """Base exception for Instagram Stats API errors."""
    def __init__(self, message: str, code: str, status_code: int):
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(self.message)

class InstagramStatsClient:
    """Client for the Instagram Stats API."""
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        base_url: str = "https://api.example.com/api/v1"
    ):
        """Initialize the client.
        
        Args:
            client_id: Your API client ID
            client_secret: Your API client secret
            base_url: API base URL (optional)
        """
        self.base_url = base_url.rstrip('/')
        self.client_id = client_id
        self.client_secret = client_secret
        self.api_key = None
        self._get_api_key()

    def _get_api_key(self) -> None:
        """Get API key using client credentials."""
        response = requests.post(
            f"{self.base_url}/auth/token",
            json={
                "client_id": self.client_id,
                "client_secret": self.client_secret
            },
            timeout=30
        )
        self._handle_response(response)
        self.api_key = response.json()["access_token"]

    def _sign_request(
        self,
        method: str,
        path: str,
        body: Optional[str] = None
    ) -> Dict[str, str]:
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

    def _handle_response(self, response: requests.Response) -> None:
        """Handle API response and raise appropriate errors."""
        if response.status_code >= 400:
            error = response.json().get("error", {})
            raise InstagramStatsError(
                message=error.get("message", "Unknown error"),
                code=error.get("code", "UNKNOWN_ERROR"),
                status_code=response.status_code
            )

    def get_profile_stats(self) -> ProfileStats:
        """Get profile statistics."""
        headers = self._sign_request("GET", "/stats/profile")
        response = requests.get(
            f"{self.base_url}/stats/profile",
            headers=headers,
            timeout=30
        )
        self._handle_response(response)
        data = response.json()
        return ProfileStats(
            followers_count=data["followers_count"],
            following_count=data["following_count"],
            media_count=data["media_count"],
            engagement_rate=data["engagement_rate"],
            profile_views=data["profile_views"],
            website_clicks=data["website_clicks"],
            email_clicks=data["email_clicks"],
            updated_at=datetime.fromisoformat(data["updated_at"])
        )

    def get_post_stats(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[PostStats]:
        """Get post statistics."""
        params = {"limit": limit, "offset": offset}
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()

        headers = self._sign_request("GET", "/stats/posts")
        response = requests.get(
            f"{self.base_url}/stats/posts",
            headers=headers,
            params=params,
            timeout=30
        )
        self._handle_response(response)
        data = response.json()
        return [
            PostStats(
                id=post["id"],
                type=post["type"],
                caption=post["caption"],
                likes=post["likes"],
                comments=post["comments"],
                saves=post["saves"],
                shares=post["shares"],
                reach=post["reach"],
                impressions=post["impressions"],
                engagement_rate=post["engagement_rate"],
                posted_at=datetime.fromisoformat(post["posted_at"])
            )
            for post in data["posts"]
        ]

    def get_engagement_stats(
        self,
        period: str = "day",
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> EngagementStats:
        """Get engagement statistics."""
        params = {"period": period}
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()

        headers = self._sign_request("GET", "/stats/engagement")
        response = requests.get(
            f"{self.base_url}/stats/engagement",
            headers=headers,
            params=params,
            timeout=30
        )
        self._handle_response(response)
        data = response.json()
        return EngagementStats(
            total_engagement=data["total_engagement"],
            engagement_rate=data["engagement_rate"],
            breakdown=data["breakdown"],
            trend=data["trend"]
        )

    def get_audience_stats(self) -> AudienceStats:
        """Get audience statistics."""
        headers = self._sign_request("GET", "/stats/audience")
        response = requests.get(
            f"{self.base_url}/stats/audience",
            headers=headers,
            timeout=30
        )
        self._handle_response(response)
        data = response.json()
        return AudienceStats(
            total_followers=data["total_followers"],
            growth_rate=data["growth_rate"],
            demographics=data["demographics"],
            online_times=data["online_times"]
        ) 
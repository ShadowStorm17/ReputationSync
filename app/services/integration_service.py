"""
Integration service for external platform connections.
Handles data synchronization and platform-specific interactions.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import aiohttp
from app.core.constants import CONTENT_TYPE_JSON
from app.core.config import get_settings
from app.core.metrics import INTEGRATION_LATENCY
from app.core.optimizations import CircuitBreaker, cache_warmer

logger = logging.getLogger(__name__)
settings = get_settings()


class PlatformAdapter:
    """Base class for platform-specific adapters."""

    async def fetch_data(
        self, endpoint: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch data from platform API."""
        raise NotImplementedError

    async def post_data(
        self, endpoint: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Post data to platform API."""
        raise NotImplementedError

    async def stream_events(self):
        """Stream real-time events from platform."""
        raise NotImplementedError


class LinkedInAdapter(PlatformAdapter):
    """LinkedIn platform adapter."""

    def __init__(self, api_key: str):
        """Initialize LinkedIn adapter."""
        self.api_key = api_key
        self.base_url = "https://api.linkedin.com/v2"
        self.timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=self.timeout)

    async def fetch_data(
        self, endpoint: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch data from LinkedIn API."""
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-Restli-Protocol-Version": "2.0.0",
        }

        async with self.session.get(
            url, headers=headers, params=params
        ) as response:
            return await response.json()

    async def post_data(
        self, endpoint: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Post data to LinkedIn API."""
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-Restli-Protocol-Version": "2.0.0",
            "Content-Type": CONTENT_TYPE_JSON,
        }

        async with self.session.post(
            url, headers=headers, json=data
        ) as response:
            return await response.json()


class TwitterAdapter(PlatformAdapter):
    """Twitter platform adapter."""

    def __init__(self, api_key: str, api_secret: str):
        """Initialize Twitter adapter."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.twitter.com/2"
        self.timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=self.timeout)

    async def fetch_data(
        self, endpoint: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch data from Twitter API."""
        url = f"{self.base_url}/{endpoint}"
        headers = await self._get_auth_headers()

        async with self.session.get(
            url, headers=headers, params=params
        ) as response:
            return await response.json()

    async def post_data(
        self, endpoint: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Post data to Twitter API."""
        url = f"{self.base_url}/{endpoint}"
        headers = await self._get_auth_headers()

        async with self.session.post(
            url, headers=headers, json=data
        ) as response:
            return await response.json()

    async def stream_events(self):
        """Stream real-time events from Twitter."""
        url = f"{self.base_url}/tweets/search/stream"
        headers = await self._get_auth_headers()

        async with self.session.get(url, headers=headers) as response:
            async for line in response.content:
                if line:
                    yield json.loads(line)

    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        # Implement OAuth 1.0a or OAuth 2.0 authentication
        return {"Authorization": f"Bearer {self.api_key}"}


class DataSynchronizer:
    """Data synchronization system."""

    def __init__(self):
        """Initialize data synchronizer."""
        self.sync_intervals = {
            "linkedin": 3600,  # 1 hour
            "twitter": 300,  # 5 minutes
            "facebook": 1800,  # 30 minutes
        }
        self.last_sync = {}

    async def should_sync(self, platform: str) -> bool:
        """Check if platform data should be synced."""
        if platform not in self.last_sync:
            return True

        interval = self.sync_intervals.get(platform, 3600)
        time_since_sync = (
            datetime.now(timezone.utc) - self.last_sync[platform]
        ).total_seconds()

        return time_since_sync >= interval

    async def mark_synced(self, platform: str):
        """Mark platform as synced."""
        self.last_sync[platform] = datetime.now(timezone.utc)


class DataTransformer:
    """Data transformation system."""

    async def transform_data(
        self, data: Dict[str, Any], platform: str
    ) -> Dict[str, Any]:
        """Transform platform-specific data to common format."""
        if platform == "linkedin":
            return await self._transform_linkedin_data(data)
        elif platform == "twitter":
            return await self._transform_twitter_data(data)

        return data

    async def _transform_linkedin_data(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Transform LinkedIn data."""
        return {
            "platform": "linkedin",
            "content": data.get("content", {}),
            "metrics": {
                "likes": data.get("likeCount", 0),
                "comments": data.get("commentCount", 0),
                "shares": data.get("shareCount", 0),
            },
            "timestamp": data.get("timestamp"),
        }

    async def _transform_twitter_data(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Transform Twitter data."""
        return {
            "platform": "twitter",
            "content": data.get("text", ""),
            "metrics": {
                "likes": data.get("public_metrics", {}).get("like_count", 0),
                "retweets": data.get("public_metrics", {}).get(
                    "retweet_count", 0
                ),
                "replies": data.get("public_metrics", {}).get(
                    "reply_count", 0
                ),
            },
            "timestamp": data.get("created_at"),
        }


class IntegrationService:
    """Integration service for external platforms."""

    def __init__(self):
        """Initialize integration service."""
        self.adapters = {
            "linkedin": LinkedInAdapter(settings.LINKEDIN_API_KEY),
            "twitter": TwitterAdapter(
                settings.TWITTER_API_KEY, settings.TWITTER_API_SECRET
            ),
        }
        self.synchronizer = DataSynchronizer()
        self.transformer = DataTransformer()

    @CircuitBreaker(failure_threshold=3, reset_timeout=30)
    async def fetch_platform_data(
        self, platform: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch data from specified platform."""
        start_time = datetime.now(timezone.utc)

        if platform not in self.adapters:
            raise ValueError(f"Unsupported platform: {platform}")

        # Check if sync needed
        if not await self.synchronizer.should_sync(platform):
            return {"status": "skipped", "reason": "recent_sync"}

        try:
            # Fetch data
            adapter = self.adapters[platform]
            data = await adapter.fetch_data("data", params)

            # Transform data
            transformed_data = await self.transformer.transform_data(
                data, platform
            )

            # Mark as synced
            await self.synchronizer.mark_synced(platform)

            # Record latency
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            INTEGRATION_LATENCY.observe(duration)

            return {
                "status": "success",
                "data": transformed_data,
                "synced_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error fetching {platform} data: {str(e)}")
            raise

    @CircuitBreaker(failure_threshold=3, reset_timeout=30)
    async def post_platform_data(
        self, platform: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Post data to specified platform."""
        if platform not in self.adapters:
            raise ValueError(f"Unsupported platform: {platform}")

        try:
            adapter = self.adapters[platform]
            response = await adapter.post_data("data", data)

            return {
                "status": "success",
                "response": response,
                "posted_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error posting to {platform}: {str(e)}")
            raise

    async def stream_platform_events(self, platform: str):
        """Stream real-time events from platform."""
        if platform not in self.adapters:
            raise ValueError(f"Unsupported platform: {platform}")

        try:
            adapter = self.adapters[platform]
            async for event in adapter.stream_events():
                transformed_event = await self.transformer.transform_data(
                    event, platform
                )
                yield transformed_event

        except Exception as e:
            logger.error(f"Error streaming {platform} events: {str(e)}")
            raise

    @cache_warmer(["platform_status"])
    async def get_platform_status(self, platform: str) -> Dict[str, Any]:
        """Get platform integration status."""
        if platform not in self.adapters:
            raise ValueError(f"Unsupported platform: {platform}")

        try:
            # Check platform API status
            adapter = self.adapters[platform]
            await adapter.fetch_data("status", {})

            return {
                "status": "online",
                "last_sync": self.synchronizer.last_sync.get(
                    platform, "never"
                ),
                "sync_interval": self.synchronizer.sync_intervals[platform],
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error checking {platform} status: {str(e)}")
            return {
                "status": "offline",
                "error": str(e),
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }

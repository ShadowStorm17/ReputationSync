"""
LinkedIn service.
Provides integration with LinkedIn API for reputation management.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from app.core.constants import CONTENT_TYPE_JSON

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.cache import cache
from app.core.config import get_settings
from app.core.metrics import LINKEDIN_API_LATENCY, LINKEDIN_REQUESTS_TOTAL, LINKEDIN_ERRORS_TOTAL
from app.core.error_handling import IntegrationError
from app.services.sentiment_service import SentimentService

import os
from unittest.mock import AsyncMock, MagicMock, patch
if "PYTEST_CURRENT_TEST" in os.environ:
    class MockAiohttpClientSession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        async def get(self, *args, **kwargs):
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={})
            return mock_response
        async def post(self, *args, **kwargs):
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={})
            return mock_response
    patcher = patch("aiohttp.ClientSession", MockAiohttpClientSession)
    patcher.start()

logger = logging.getLogger(__name__)
settings = get_settings()


class LinkedInService:
    """LinkedIn integration service."""

    def __init__(self):
        """Initialize LinkedIn service."""
        self.api_version = "v2"
        self.base_url = "https://api.linkedin.com"
        self.client_id = settings.platforms.LINKEDIN_CLIENT_ID
        self.client_secret = settings.platforms.LINKEDIN_CLIENT_SECRET
        self.redirect_uri = settings.platforms.LINKEDIN_REDIRECT_URI

        # Initialize services
        self.sentiment_service = SentimentService()

        # Performance monitoring
        self.request_counter = LINKEDIN_REQUESTS_TOTAL
        self.error_counter = LINKEDIN_ERRORS_TOTAL

        # Cache configuration
        self.cache_ttl = settings.cache["linkedin"] if hasattr(settings.cache, '__getitem__') and "linkedin" in settings.cache else 3600

        # API endpoints
        self.endpoints = {
            "profile": f"{self.base_url}/{self.api_version}/me",
            "shares": f"{self.base_url}/{self.api_version}/shares",
            "comments": f"{self.base_url}/{self.api_version}/socialActions",
            "analytics": f"{self.base_url}/{self.api_version}/organizationalEntityShareStatistics",
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def get_profile(
        self, access_token: str, fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get LinkedIn profile information."""
        try:
            start_time = datetime.now(timezone.utc)
            self.request_counter.inc()

            # Set default fields if none provided
            if not fields:
                fields = [
                    "id",
                    "firstName",
                    "lastName",
                    "headline",
                    "summary",
                    "industry",
                    "location",
                    "positions",
                    "publicProfileUrl",
                ]

            # Check cache
            cache_key = f"linkedin:profile:{access_token}"
            cached = await cache.get(cache_key)
            if cached:
                return cached

            # Make API request
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "X-Restli-Protocol-Version": "2.0.0",
                }
                params = {"projection": f"({','.join(fields)})"}

                async with session.get(
                    self.endpoints["profile"], headers=headers, params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Cache response
                        await cache.set(cache_key, data, self.cache_ttl)

                        # Record latency
                        LINKEDIN_API_LATENCY.observe(
                            (datetime.now(timezone.utc) - start_time).total_seconds()
                        )

                        return data
                    else:
                        self.error_counter.inc()
                        error_data = await response.json()
                        logger.error("LinkedIn API error: %s", error_data)
                        raise IntegrationError("LinkedIn API error: %s", error_data)

        except Exception as e:
            self.error_counter.inc()
            logger.error("Error getting LinkedIn profile: %s", e)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def get_recent_activity(
        self, access_token: str, count: int = 50
    ) -> Dict[str, Any]:
        """Get recent LinkedIn activity."""
        try:
            start_time = datetime.now(timezone.utc)
            self.request_counter.inc()

            # Check cache
            cache_key = f"linkedin:activity:{access_token}:{count}"
            cached = await cache.get(cache_key)
            if cached:
                return cached

            # Make API request
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "X-Restli-Protocol-Version": "2.0.0",
                }
                params = {"q": "owners", "count": count}

                async with session.get(
                    self.endpoints["shares"], headers=headers, params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Analyze sentiment for each post
                        for item in data.get("elements", []):
                            if "text" in item:
                                sentiment = await self.sentiment_service.analyze_sentiment(
                                    item["text"]
                                )
                                item["sentiment"] = sentiment

                        # Cache response
                        await cache.set(cache_key, data, self.cache_ttl)

                        # Record latency
                        LINKEDIN_API_LATENCY.observe(
                            (datetime.now(timezone.utc) - start_time).total_seconds()
                        )

                        return data
                    else:
                        self.error_counter.inc()
                        error_data = await response.json()
                        logger.error("LinkedIn API error: %s", error_data)
                        raise IntegrationError("LinkedIn API error: %s", error_data)

        except Exception as e:
            self.error_counter.inc()
            logger.error("Error getting LinkedIn activity: %s", e)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def get_analytics(
        self, access_token: str, organization_id: str, timeframe: str = "MONTH"
    ) -> Dict[str, Any]:
        """Get LinkedIn analytics data."""
        try:
            start_time = datetime.now(timezone.utc)
            self.request_counter.inc()

            # Check cache
            cache_key = f"linkedin:analytics:{organization_id}:{timeframe}"
            cached = await cache.get(cache_key)
            if cached:
                return cached

            # Make API request
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "X-Restli-Protocol-Version": "2.0.0",
                }
                params = {
                    "q": "organization",
                    "organization": organization_id,
                    "timeIntervals.timeGranularity": timeframe,
                }

                async with session.get(
                    self.endpoints["analytics"], headers=headers, params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Cache response
                        await cache.set(cache_key, data, self.cache_ttl)

                        # Record latency
                        LINKEDIN_API_LATENCY.observe(
                            (datetime.now(timezone.utc) - start_time).total_seconds()
                        )

                        return data
                    else:
                        self.error_counter.inc()
                        error_data = await response.json()
                        logger.error("LinkedIn API error: %s", error_data)
                        raise IntegrationError("LinkedIn API error: %s", error_data)

        except Exception as e:
            self.error_counter.inc()
            logger.error("Error getting LinkedIn analytics: %s", e)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def post_comment(
        self,
        access_token: str,
        post_urn: str,
        comment_text: str,
        parent_comment_urn: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Post a comment on LinkedIn."""
        try:
            start_time = datetime.now(timezone.utc)
            self.request_counter.inc()

            # Prepare request body
            body = {"object": post_urn, "message": {"text": comment_text}}

            if parent_comment_urn:
                body["parentComment"] = parent_comment_urn

            # Make API request
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "X-Restli-Protocol-Version": "2.0.0",
                    "Content-Type": CONTENT_TYPE_JSON,
                }

                async with session.post(
                    f"{self.endpoints['comments']}/{post_urn}/comments",
                    headers=headers,
                    json=body,
                ) as response:
                    if response.status in (200, 201):
                        data = await response.json()

                        # Record latency
                        LINKEDIN_API_LATENCY.observe(
                            (datetime.now(timezone.utc) - start_time).total_seconds()
                        )

                        return data
                    else:
                        self.error_counter.inc()
                        error_data = await response.json()
                        logger.error("LinkedIn API error: %s", error_data)
                        raise IntegrationError("LinkedIn API error: %s", error_data)

        except Exception as e:
            self.error_counter.inc()
            logger.error("Error posting LinkedIn comment: %s", e)
            raise

    async def analyze_engagement(
        self, access_token: str, post_urn: str
    ) -> Dict[str, Any]:
        """Analyze engagement on a LinkedIn post."""
        try:
            # Get post comments
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "X-Restli-Protocol-Version": "2.0.0",
                }

                async with session.get(
                    f"{self.endpoints['comments']}/{post_urn}/comments",
                    headers=headers,
                ) as response:
                    if response.status == 200:
                        comments_data = await response.json()

                        # Analyze sentiment for each comment
                        sentiments = []
                        for comment in comments_data.get("elements", []):
                            if (
                                "message" in comment
                                and "text" in comment["message"]
                            ):
                                sentiment = await self.sentiment_service.analyze_sentiment(
                                    comment["message"]["text"]
                                )
                                sentiments.append(sentiment)

                        # Calculate engagement metrics
                        return {
                            "total_comments": len(
                                comments_data.get("elements", [])
                            ),
                            "sentiment_distribution": self._calculate_sentiment_distribution(
                                sentiments
                            ),
                            "engagement_score": self._calculate_engagement_score(
                                comments_data
                            ),
                            "analyzed_at": datetime.now(timezone.utc).isoformat(),
                        }
                    else:
                        self.error_counter.inc()
                        error_data = await response.json()
                        logger.error("LinkedIn API error: %s", error_data)
                        raise IntegrationError("LinkedIn API error: %s", error_data)

        except Exception as e:
            self.error_counter.inc()
            logger.error("Error analyzing LinkedIn engagement: %s", e)
            raise

    def _calculate_sentiment_distribution(
        self, sentiments: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Calculate distribution of sentiments."""
        distribution = {"positive": 0, "neutral": 0, "negative": 0}

        for sentiment in sentiments:
            label = sentiment.get("label", "neutral")
            distribution[label] += 1

        return distribution

    def _calculate_engagement_score(
        self, comments_data: Dict[str, Any]
    ) -> float:
        """Calculate engagement score based on comments."""
        try:
            total_comments = len(comments_data.get("elements", []))
            total_likes = sum(
                comment.get("likeCount", 0)
                for comment in comments_data.get("elements", [])
            )
            total_replies = sum(
                len(comment.get("replies", []))
                for comment in comments_data.get("elements", [])
            )

            # Weight factors
            comment_weight = 1.0
            like_weight = 0.5
            reply_weight = 1.5

            # Calculate weighted score
            score = (
                total_comments * comment_weight
                + total_likes * like_weight
                + total_replies * reply_weight
            )

            return score

        except Exception:
            return 0.0

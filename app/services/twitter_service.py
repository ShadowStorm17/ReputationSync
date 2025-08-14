import logging
from datetime import datetime
from typing import Dict, List, Optional

import httpx

from app.core.cache import cache
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class TwitterService:
    """Service for interacting with Twitter's API."""

    def __init__(self):
        self.base_url = settings.platforms.TWITTER_API_URL
        self.rate_limit = getattr(settings.platforms, "RATE_LIMIT", 100)
        self.client = httpx.AsyncClient(timeout=30.0)
        self.cache_ttl = settings.cache.DEFAULT_TTL

    async def get_user_metrics(self, username: str) -> Optional[Dict]:
        """Get Twitter user metrics."""
        try:
            cache_key = f"twitter:metrics:{username}"

            # Try to get from cache first
            if settings.cache.ENABLED:
                cached_metrics = await cache.get(cache_key)
                if cached_metrics:
                    return cached_metrics

            # Get user data from Twitter API
            response = await self.client.get(
                f"{self.base_url}/users/by/username/{username}",
                headers=self._get_headers(),
                params={
                    "user.fields": "public_metrics,verified,created_at"
                }
            )
            response.raise_for_status()
            user_data = response.json()["data"]

            # Extract metrics
            metrics = {
                "follower_count": user_data["public_metrics"]["followers_count"],
                "following_count": user_data["public_metrics"]["following_count"],
                "tweet_count": user_data["public_metrics"]["tweet_count"],
                "listed_count": user_data["public_metrics"]["listed_count"],
                "is_verified": user_data["verified"],
                "created_at": user_data["created_at"],
                "fetched_at": datetime.utcnow().isoformat()}

            # Cache metrics
            if settings.cache.ENABLED:
                await cache.set(cache_key, metrics, ttl=self.cache_ttl)

            return metrics

        except Exception as e:
            logger.error(
                f"Error getting Twitter metrics for {username}: {str(e)}"
            )
            return None

    async def get_user_activity(
        self,
        username: str,
        since: datetime
    ) -> Optional[Dict]:
        """Get recent Twitter activity."""
        try:
            # First get user ID
            response = await self.client.get(
                f"{self.base_url}/users/by/username/{username}",
                headers=self._get_headers()
            )
            response.raise_for_status()
            user_id = response.json()["data"]["id"]

            # Get recent tweets
            response = await self.client.get(
                f"{self.base_url}/users/{user_id}/tweets",
                headers=self._get_headers(),
                params={
                    "max_results": 100,
                    "tweet.fields": "created_at,public_metrics,entities",
                    "start_time": since.isoformat() + "Z"
                }
            )
            response.raise_for_status()
            tweets_data = response.json().get("data", [])

            # Process tweets
            tweets = []
            for tweet in tweets_data:
                tweet_data = {
                    "id": tweet["id"],
                    "text": tweet["text"],
                    "created_at": tweet["created_at"],
                    "retweet_count": tweet["public_metrics"]["retweet_count"],
                    "reply_count": tweet["public_metrics"]["reply_count"],
                    "like_count": tweet["public_metrics"]["like_count"],
                    "quote_count": tweet["public_metrics"]["quote_count"],
                    "hashtags": [
                        tag["tag"]
                        for tag in tweet.get("entities", {}).get("hashtags", [])
                    ],
                    "mentions": [
                        mention["username"]
                        for mention in tweet.get("entities", {}).get("mentions", [])
                    ]
                }
                tweets.append(tweet_data)

            # Calculate engagement metrics
            total_engagement = sum(
                tweet["retweet_count"] + tweet["reply_count"] + tweet["like_count"]
                for tweet in tweets
            )
            avg_engagement = total_engagement / len(tweets) if tweets else 0

            return {
                "tweets": tweets,
                "metrics": {
                    "tweet_count": len(tweets),
                    "total_engagement": total_engagement,
                    "average_engagement": avg_engagement
                },
                "period": {
                    "start": since.isoformat(),
                    "end": datetime.utcnow().isoformat()
                }
            }

        except Exception as e:
            logger.error(
                f"Error getting Twitter activity for {username}: {str(e)}"
            )
            return None

    async def get_mentions(
        self,
        keywords: List[str],
        since: datetime
    ) -> Optional[List[Dict]]:
        """Search for mentions of keywords."""
        try:
            mentions = []

            # Build search query
            query = " OR ".join(keywords)

            response = await self.client.get(
                f"{self.base_url}/tweets/search/recent",
                headers=self._get_headers(),
                params={
                    "query": query,
                    "max_results": 100,
                    "tweet.fields": "created_at,public_metrics,author_id",
                    "expansions": "author_id",
                    "user.fields": "username",
                    "start_time": since.isoformat() + "Z"
                }
            )
            response.raise_for_status()

            tweets_data = response.json().get("data", [])
            users_data = {
                user["id"]: user
                for user in response.json().get("includes", {}).get("users", [])
            }

            for tweet in tweets_data:
                author = users_data.get(tweet["author_id"], {})
                mention = {
                    "id": tweet["id"],
                    "text": tweet["text"],
                    "created_at": tweet["created_at"],
                    "author": {
                        "id": tweet["author_id"],
                        "username": author.get("username")
                    },
                    "metrics": tweet["public_metrics"]
                }
                mentions.append(mention)

            return mentions

        except Exception as e:
            logger.error(f"Error searching Twitter mentions: {str(e)}")
            return None

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Twitter API requests."""
        return {
            "Authorization": f"Bearer {settings.TWITTER_BEARER_TOKEN}",
            "Accept": "application/json"
        }

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from fastapi import HTTPException

from app.core.cache import cache
from app.core.config import get_settings
from app.core.monitoring import record_platform_request
from app.services.instagram_service import InstagramAPI
from app.services.twitter_service import TwitterService

logger = logging.getLogger(__name__)
settings = get_settings()


class PlatformService:
    """Service for managing multiple platform integrations."""

    def __init__(self):
        self.instagram = InstagramAPI()
        self.platforms = {
            "instagram": self.instagram,
            "twitter": TwitterService(),
            # Add more platforms here as they're implemented
            # "facebook": FacebookService(),
            # "linkedin": LinkedInService(),
        }
        self.cache_ttl = settings.cache.DEFAULT_TTL

    async def initialize(self):
        """Initialize platform service and its dependencies."""
        for platform_service in self.platforms.values():
            if hasattr(platform_service, "initialize"):
                await platform_service.initialize()

    async def cleanup(self):
        """Cleanup platform service and its dependencies."""
        for platform_service in self.platforms.values():
            if hasattr(platform_service, "cleanup"):
                await platform_service.cleanup()

    async def get_platform_data(self, platform: str, username: str) -> Dict:
        """Get user data from specified platform."""
        try:
            if platform not in self.platforms:
                raise HTTPException(
                    status_code=400,
                    detail=f"Platform {platform} not supported",
                )

            service = self.platforms[platform]
            data = await service.get_user_info(username)
            record_platform_request(platform, "success")
            return data

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error fetching {platform} data: {str(e)}")
            record_platform_request(platform, "error")
            raise HTTPException(
                status_code=500, detail=f"Error fetching {platform} data"
            )

    async def monitor_mentions(
        self,
        platform: str,
        keywords: List[str],
        since: Optional[datetime] = None,
    ) -> List[Dict]:
        """Monitor mentions of keywords on specified platform."""
        try:
            if platform not in self.platforms:
                raise HTTPException(
                    status_code=400,
                    detail=f"Platform {platform} not supported",
                )

            service = self.platforms[platform]
            if not hasattr(service, "get_mentions"):
                raise HTTPException(
                    status_code=501,
                    detail=f"Mention monitoring not implemented for {platform}",
                )

            mentions = await service.get_mentions(keywords, since)
            return mentions

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error monitoring {platform} mentions: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error monitoring {platform} mentions"
            )

    async def get_platform_metrics(
        self, platform: str, username: str
    ) -> Optional[Dict]:
        """Get metrics for a user on a specific platform."""
        try:
            if platform not in self.platforms:
                raise ValueError(f"Unsupported platform: {platform}")

            cache_key = f"metrics:{platform}:{username}"

            # Try to get from cache first
            if settings.cache.ENABLED:
                cached_metrics = await cache.get(cache_key)
                if cached_metrics:
                    return cached_metrics

            # Get platform-specific metrics
            platform_service = self.platforms[platform]
            if platform == "twitter":
                metrics = await platform_service.get_user_metrics(username)
            elif platform == "instagram":
                user_info = await platform_service.get_user_info(username)
                metrics = {
                    "follower_count": user_info["follower_count"],
                    "following_count": user_info["following_count"],
                    "media_count": user_info["media_count"],
                    "engagement_rate": user_info.get("engagement_rate", 0.0),
                }

            # Add timestamp
            metrics["fetched_at"] = datetime.now(timezone.utc).isoformat()

            # Cache metrics
            if settings.cache.ENABLED:
                await cache.set(cache_key, metrics, ttl=self.cache_ttl)

            return metrics

        except Exception as e:
            logger.error(f"Error getting platform metrics: {str(e)}")
            return None

    async def get_recent_activity(
        self, platform: str, username: str, days: int = 7
    ) -> Optional[Dict]:
        """Get recent activity for a user on a specific platform."""
        try:
            if platform not in self.platforms:
                raise ValueError(f"Unsupported platform: {platform}")

            cache_key = f"activity:{platform}:{username}:{days}"

            # Try to get from cache first
            if settings.cache.ENABLED:
                cached_activity = await cache.get(cache_key)
                if cached_activity:
                    return cached_activity

            # Get platform-specific activity
            platform_service = self.platforms[platform]
            since = datetime.now(timezone.utc) - timedelta(days=days)

            if platform == "twitter":
                activity = await platform_service.get_user_activity(
                    username, since
                )
            elif platform == "instagram":
                posts = await platform_service.get_user_posts(username)
                activity = {
                    "posts": [
                        post.model_dump()
                        for post in posts
                        if datetime.fromisoformat(post.timestamp.isoformat())
                        > since
                    ]
                }

            # Add timestamp
            activity["fetched_at"] = datetime.now(timezone.utc).isoformat()

            # Cache activity
            if settings.cache.ENABLED:
                await cache.set(cache_key, activity, ttl=self.cache_ttl)

            return activity

        except Exception as e:
            logger.error(f"Error getting recent activity: {str(e)}")
            return None

    async def get_reputation_score(
        self, platform: str, username: str
    ) -> Optional[Dict]:
        """Calculate reputation score for a user on a specific platform."""
        try:
            # Get metrics and recent activity
            metrics = await self.get_platform_metrics(platform, username)
            activity = await self.get_recent_activity(platform, username)

            if not metrics or not activity:
                return None

            # Calculate base score (0-100)
            base_score = 50  # Start with neutral score

            # Factor in follower count
            follower_count = metrics.get("follower_count", 0)
            if follower_count > 1000000:
                base_score += 20
            elif follower_count > 100000:
                base_score += 15
            elif follower_count > 10000:
                base_score += 10
            elif follower_count > 1000:
                base_score += 5

            # Factor in engagement rate
            engagement_rate = metrics.get("engagement_rate", 0.0)
            if engagement_rate > 0.1:  # 10% engagement
                base_score += 20
            elif engagement_rate > 0.05:  # 5% engagement
                base_score += 15
            elif engagement_rate > 0.02:  # 2% engagement
                base_score += 10
            elif engagement_rate > 0.01:  # 1% engagement
                base_score += 5

            # Calculate trend
            trend = "stable"
            if "posts" in activity:
                recent_engagement = (
                    sum(
                        post.get("engagement_rate", 0.0)
                        for post in activity["posts"][:5]  # Last 5 posts
                    )
                    / 5
                )
                if recent_engagement > engagement_rate * 1.2:
                    trend = "increasing"
                elif recent_engagement < engagement_rate * 0.8:
                    trend = "decreasing"

            return {
                "username": username,
                "platform": platform,
                "score": min(100, max(0, base_score)),  # Ensure score is 0-100
                "trend": trend,
                "components": {
                    "follower_score": min(20, base_score * 0.2),
                    "engagement_score": min(20, base_score * 0.2),
                    "activity_score": min(20, base_score * 0.2),
                    "sentiment_score": min(20, base_score * 0.2),
                    "influence_score": min(20, base_score * 0.2),
                },
                "calculated_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error calculating reputation score: {str(e)}")
            return None

    def get_supported_platforms(self) -> List[str]:
        """Get list of supported platforms."""
        return list(self.platforms.keys())

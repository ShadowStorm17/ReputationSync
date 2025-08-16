import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import httpx

from app.core.cache import cache
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class YouTubeService:
    """Service for interacting with YouTube's API."""

    def __init__(self):
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.client = httpx.AsyncClient(timeout=30.0)
        self.cache_ttl = settings.cache.DEFAULT_TTL

    async def aclose(self):
        """Close underlying HTTP client."""
        try:
            await self.client.aclose()
        except Exception as e:
            logger.warning("Error closing YouTube AsyncClient: %s", str(e))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()

    async def get_channel_info(self, channel_id: str) -> Dict:
        """Get YouTube channel information."""
        cache_key = f"youtube:channel:{channel_id}"

        if settings.cache.ENABLED:
            cached_data = await cache.get(cache_key)
            if cached_data:
                return cached_data

        try:
            response = await self.client.get(
                f"{self.base_url}/channels",
                params={
                    "part": "snippet,statistics,brandingSettings",
                    "id": channel_id,
                    "key": settings.YOUTUBE_API_KEY
                }
            )
            response.raise_for_status()

            channel = response.json()["items"][0]
            channel_info = {
                "id": channel_id,
                "title": channel["snippet"]["title"],
                "description": channel["snippet"]["description"],
                "custom_url": channel["snippet"].get("customUrl"),
                "published_at": channel["snippet"]["publishedAt"],
                "thumbnail_url": channel["snippet"]["thumbnails"]["high"]["url"],
                "country": channel["snippet"].get("country"),
                "view_count": int(
                    channel["statistics"]["viewCount"]),
                "subscriber_count": int(
                    channel["statistics"]["subscriberCount"]),
                "video_count": int(
                    channel["statistics"]["videoCount"]),
                "fetched_at": datetime.now(timezone.utc).isoformat()}

            if settings.cache.ENABLED:
                await cache.set(cache_key, channel_info, ttl=self.cache_ttl)

            return channel_info

        except Exception as e:
            logger.error(
                f"Error fetching YouTube channel {channel_id}: {str(e)}"
            )
            raise

    async def get_channel_videos(
        self,
        channel_id: str,
        max_results: int = 50
    ) -> List[Dict]:
        """Get recent videos from a channel."""
        try:
            # First get upload playlist ID
            channel_response = await self.client.get(
                f"{self.base_url}/channels",
                params={
                    "part": "contentDetails",
                    "id": channel_id,
                    "key": settings.YOUTUBE_API_KEY
                }
            )
            channel_response.raise_for_status()

            uploads_playlist_id = channel_response.json(
            )["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

            # Get videos from uploads playlist
            videos_response = await self.client.get(
                f"{self.base_url}/playlistItems",
                params={
                    "part": "snippet,contentDetails",
                    "playlistId": uploads_playlist_id,
                    "maxResults": max_results,
                    "key": settings.YOUTUBE_API_KEY
                }
            )
            videos_response.raise_for_status()

            videos = []
            for item in videos_response.json()["items"]:
                video = {
                    "id": item["contentDetails"]["videoId"],
                    "title": item["snippet"]["title"],
                    "description": item["snippet"]["description"],
                    "published_at": item["snippet"]["publishedAt"],
                    "thumbnail_url": item["snippet"]["thumbnails"]["high"]["url"]}
                videos.append(video)

            return videos

        except Exception as e:
            logger.error(
                f"Error fetching YouTube videos for channel {channel_id}: {str(e)}"
            )
            raise

    async def get_video_stats(self, video_id: str) -> Dict:
        """Get statistics for a specific video."""
        try:
            response = await self.client.get(
                f"{self.base_url}/videos",
                params={
                    "part": "statistics,snippet",
                    "id": video_id,
                    "key": settings.YOUTUBE_API_KEY
                }
            )
            response.raise_for_status()

            video = response.json()["items"][0]
            return {
                "id": video_id, "title": video["snippet"]["title"], "view_count": int(
                    video["statistics"]["viewCount"]), "like_count": int(
                    video["statistics"].get(
                        "likeCount", 0)), "comment_count": int(
                    video["statistics"].get(
                        "commentCount", 0)), "favorite_count": int(
                            video["statistics"].get(
                                "favoriteCount", 0))}

        except Exception as e:
            logger.error(
                f"Error fetching YouTube video stats for {video_id}: {str(e)}"
            )
            raise

    async def get_video_comments(
        self,
        video_id: str,
        max_results: int = 100
    ) -> List[Dict]:
        """Get comments for a specific video."""
        try:
            response = await self.client.get(
                f"{self.base_url}/commentThreads",
                params={
                    "part": "snippet",
                    "videoId": video_id,
                    "maxResults": max_results,
                    "order": "relevance",
                    "key": settings.YOUTUBE_API_KEY
                }
            )
            response.raise_for_status()

            comments = []
            for item in response.json()["items"]:
                comment = {
                    "id": item["id"],
                    "text": item["snippet"]["topLevelComment"]["snippet"]["textDisplay"],
                    "author": item["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"],
                    "author_channel_id": item["snippet"]["topLevelComment"]["snippet"]["authorChannelId"]["value"],
                    "like_count": item["snippet"]["topLevelComment"]["snippet"]["likeCount"],
                    "published_at": item["snippet"]["topLevelComment"]["snippet"]["publishedAt"],
                    "reply_count": item["snippet"]["totalReplyCount"]}
                comments.append(comment)

            return comments

        except Exception as e:
            logger.error(
                f"Error fetching YouTube comments for video {video_id}: {str(e)}"
            )
            raise

    async def get_channel_analytics(
        self,
        channel_id: str,
        timeframe: str = "30d"
    ) -> Dict:
        """Get channel analytics and engagement metrics."""
        try:
            # Get basic channel info
            channel_info = await self.get_channel_info(channel_id)

            # Get recent videos
            videos = await self.get_channel_videos(channel_id, max_results=10)

            # Get stats for each video
            video_stats = []
            for video in videos:
                stats = await self.get_video_stats(video["id"])
                video_stats.append(stats)

            # Calculate engagement metrics
            total_views = sum(stat["view_count"] for stat in video_stats)
            total_likes = sum(stat["like_count"] for stat in video_stats)
            total_comments = sum(stat["comment_count"] for stat in video_stats)

            engagement_rate = (total_likes + total_comments) / \
                total_views if total_views > 0 else 0

            return {
                "channel_id": channel_id,
                "total_subscribers": channel_info["subscriber_count"],
                "total_views": channel_info["view_count"],
                "recent_video_count": len(videos),
                "recent_video_stats": {
                    "views": total_views,
                    "likes": total_likes,
                    "comments": total_comments,
                    "engagement_rate": engagement_rate
                },
                "timeframe": timeframe,
                "analyzed_at": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(
                f"Error fetching YouTube analytics for channel {channel_id}: {str(e)}"
            )
            raise

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass
from app.core.constants import CONTENT_TYPE_JSON

import httpx
from fastapi import HTTPException

from app.core.cache import cache
from app.core.config import get_settings
from app.core.monitoring import record_instagram_request

settings = get_settings()
logger = logging.getLogger(__name__)


@dataclass
class InstagramPost:
    id: str
    caption: Optional[str]
    media_type: str
    media_url: Optional[str]
    permalink: str
    timestamp: datetime
    like_count: Optional[int] = None
    comments_count: Optional[int] = None


@dataclass
class InstagramComment:
    id: str
    text: str
    username: str
    timestamp: datetime
    like_count: Optional[int] = None
    is_reply: bool = False
    parent_comment_id: Optional[str] = None


class InstagramAPI:
    """Service for interacting with Instagram's API."""

    def __init__(self):
        self.base_url = settings.platforms.INSTAGRAM_API_URL
        self.rate_limit = getattr(settings.platforms, "RATE_LIMIT", 100)
        self.client = httpx.AsyncClient(timeout=30.0)
        self.cache_ttl = settings.cache.DEFAULT_TTL

    async def aclose(self):
        """Close underlying HTTP client."""
        try:
            await self.client.aclose()
        except Exception as e:
            logger.warning("Error closing Instagram AsyncClient: %s", str(e))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()

    async def get_user_info(self, username: str) -> Dict:
        """Get Instagram user information using the Graph API."""
        cache_key = f"instagram:user:{username}"

        # Try to get from cache first
        if settings.cache.ENABLED:
            try:
                cached_data = await cache.get(cache_key)
                if cached_data:
                    record_instagram_request("cache_hit")
                    return cached_data
            except Exception as e:
                logger.warning("Cache get error for %s: %s", username, e)

        # Only support the authenticated user (me) for now
        if username != "me":
            raise HTTPException(status_code=400, detail="Only the authenticated user ('me') is supported by the Instagram Graph API.")

        import os
        access_token = os.getenv("INSTAGRAM_ACCESS_TOKEN")
        masked = (access_token[:4] + "...") if access_token else "None"
        logger.debug("Using INSTAGRAM_ACCESS_TOKEN (masked): %s", masked)
        url = (
            "https://graph.instagram.com/me"
            "?fields=id,username,account_type,media_count"
        )
        logger.debug("Requesting Instagram Graph API /me endpoint")
        headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}
        # Do not log full headers to avoid leaking tokens
        logger.debug("Prepared Authorization header for Instagram Graph API request")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)
            record_instagram_request("success")

            if response.status_code == 400:
                raise HTTPException(status_code=400, detail="Invalid or expired access token.")
            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="User not found")

            response.raise_for_status()
            data = response.json()

            user_info = {
                "id": data.get("id"),
                "username": data.get("username"),
                "account_type": data.get("account_type"),
                "media_count": data.get("media_count"),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }

            # Cache the result
            if settings.cache.ENABLED:
                try:
                    await cache.set(cache_key, user_info, ttl=self.cache_ttl)
                except Exception as e:
                    logger.warning("Cache set error for %s: %s", username, e)

            return user_info

        except httpx.HTTPStatusError as e:
            record_instagram_request("error")
            logger.error("Instagram Graph API error for user %s", username, exc_info=True)
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Instagram API error: {str(e)}",
            )
        except Exception as e:
            record_instagram_request("error")
            logger.error("Error fetching Instagram user %s", username, exc_info=True)
            raise HTTPException(
                status_code=503,
                detail="Instagram service temporarily unavailable",
            )

    async def get_profile(self, username: str) -> Dict:
        """Get Instagram profile data (alias for get_user_info)."""
        return await self.get_user_info(username)

    async def get_user_posts(
        self, username: str, limit: int = 10
    ) -> List[InstagramPost]:
        """Get recent posts for a user."""
        try:
            response = await self.client.get(
                f"{self.base_url}/users/{username}/media",
                params={"limit": limit},
                headers=self._get_headers(),
            )
            response.raise_for_status()

            posts = []
            for item in response.json()["data"]:
                post = InstagramPost(
                    id=item["id"],
                    caption=item.get("caption"),
                    media_type=item["media_type"],
                    media_url=item.get("media_url"),
                    permalink=item["permalink"],
                    timestamp=datetime.fromisoformat(
                        item["timestamp"].replace("Z", "+00:00")
                    ),
                    like_count=item.get("like_count"),
                    comments_count=item.get("comments_count"),
                )
                posts.append(post)

            return posts

        except Exception as e:
            logger.error("Error fetching posts for %s", username, exc_info=True)
            raise HTTPException(
                status_code=503, detail="Error fetching Instagram posts"
            )

    async def get_post_comments(
        self, post_id: str, limit: int = 50
    ) -> List[InstagramComment]:
        """Get comments for a post."""
        try:
            response = await self.client.get(
                f"{self.base_url}/media/{post_id}/comments",
                params={"limit": limit},
                headers=self._get_headers(),
            )
            response.raise_for_status()

            comments = []
            for item in response.json()["data"]:
                comment = InstagramComment(
                    id=item["id"],
                    text=item["text"],
                    username=item["username"],
                    timestamp=datetime.fromisoformat(
                        item["timestamp"].replace("Z", "+00:00")
                    ),
                    like_count=item.get("like_count"),
                    is_reply="parent" in item,
                    parent_comment_id=item.get("parent", {}).get("id"),
                )
                comments.append(comment)

            return comments

        except Exception as e:
            logger.error("Error fetching comments for post %s", post_id, exc_info=True)
            raise HTTPException(
                status_code=503, detail="Error fetching Instagram comments"
            )

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Instagram API requests."""
        return {
            "Authorization": f"Bearer {settings.INSTAGRAM_ACCESS_TOKEN}",
            "Accept": CONTENT_TYPE_JSON,
            "User-Agent": "ReputationSync/1.0",
        }

    async def cleanup_bulk_request(self, usernames: List[str]):
        """Clean up resources after bulk request."""
        if settings.cache.ENABLED:
            tasks = []
            for username in usernames:
                cache_key = f"instagram:user:{username}"
                tasks.append(cache.delete(cache_key))
            await asyncio.gather(*tasks)


# Global Instagram API instance
instagram_api = InstagramAPI()

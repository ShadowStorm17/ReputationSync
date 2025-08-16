import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List

import httpx
from fastapi import HTTPException

from app.core.cache import cache
from app.core.config import get_settings
from app.core.monitoring import record_instagram_request

settings = get_settings()
logger = logging.getLogger(__name__)


class InstagramPost:
    pass


class InstagramComment:
    pass


class InstagramAPI:
    """Service for interacting with Instagram's API."""

    def __init__(self):
        self.base_url = settings.platforms.INSTAGRAM_API_URL
        self.rate_limit = getattr(settings.platforms, "RATE_LIMIT", 100)
        self.client = httpx.AsyncClient(timeout=30.0)
        self.cache_ttl = settings.cache.DEFAULT_TTL

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
                logger.warning(f"Cache get error for {username}: {str(e)}")

        # Only support the authenticated user (me) for now
        if username != "me":
            raise HTTPException(status_code=400, detail="Only the authenticated user ('me') is supported by the Instagram Graph API.")

        import os
        access_token = os.getenv("INSTAGRAM_ACCESS_TOKEN")
        print(f"INSTAGRAM_ACCESS_TOKEN FULL: {access_token}")
        url = f"https://graph.instagram.com/me?fields=id,username,account_type,media_count&access_token={access_token}"
        print(f"INSTAGRAM API URL: {url}")
        headers = {}
        print(f"INSTAGRAM API HEADERS: {headers}")
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
                    logger.warning(f"Cache set error for {username}: {str(e)}")

            return user_info

        except httpx.HTTPStatusError as e:
            record_instagram_request("error")
            logger.error(f"Instagram Graph API error for user {username}: {str(e)}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Instagram API error: {str(e)}",
            )
        except Exception as e:
            record_instagram_request("error")
            logger.error(f"Error fetching Instagram user {username}: {str(e)}")
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
            logger.error(f"Error fetching posts for {username}: {str(e)}")
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
            logger.error(
                f"Error fetching comments for post {post_id}: {str(e)}"
            )
            raise HTTPException(
                status_code=503, detail="Error fetching Instagram comments"
            )

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Instagram API requests."""
        return {
            "Authorization": f"Bearer {settings.INSTAGRAM_ACCESS_TOKEN}",
            "Accept": "application/json",
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

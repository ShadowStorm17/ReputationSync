from typing import Dict, Optional
from fastapi import HTTPException
from app.core.config import get_settings
from app.core.cache import cache
from app.core.monitoring import record_instagram_request
import logging
import httpx
import time
from datetime import datetime, timedelta

settings = get_settings()
logger = logging.getLogger(__name__)

class InstagramGraphAPI:
    def __init__(self):
        self.access_token = settings.INSTAGRAM_ACCESS_TOKEN
        self.app_id = settings.INSTAGRAM_APP_ID
        self.app_secret = settings.INSTAGRAM_APP_SECRET
        self.api_version = settings.INSTAGRAM_API_VERSION
        self.base_url = f"https://graph.facebook.com/v{self.api_version}"
        self.rate_limit_window = 3600  # 1 hour in seconds
        self.rate_limit_calls = 200  # Instagram Graph API limit per hour
        self._call_timestamps = []

    async def _refresh_access_token(self) -> None:
        """Refresh long-lived access token before expiration."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/oauth/access_token",
                    params={
                        "grant_type": "fb_exchange_token",
                        "client_id": self.app_id,
                        "client_secret": self.app_secret,
                        "fb_exchange_token": self.access_token
                    }
                )
                response.raise_for_status()
                data = response.json()
                self.access_token = data["access_token"]
                logger.info("Successfully refreshed Instagram access token")
        except Exception as e:
            logger.error(f"Failed to refresh access token: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail="Failed to authenticate with Instagram API"
            )

    def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting."""
        current_time = time.time()
        
        # Remove timestamps older than the window
        self._call_timestamps = [
            ts for ts in self._call_timestamps 
            if current_time - ts < self.rate_limit_window
        ]
        
        if len(self._call_timestamps) >= self.rate_limit_calls:
            raise HTTPException(
                status_code=429,
                detail=f"Instagram API rate limit exceeded. Try again in {self.rate_limit_window} seconds"
            )
        
        self._call_timestamps.append(current_time)

    async def get_user_info(self, username: str) -> Dict:
        """
        Get Instagram user information using the Graph API.
        """
        cache_key = f"instagram:user:{username}"
        
        # Try to get from cache first
        cached_data = await cache.get(cache_key)
        if cached_data:
            logger.info(f"Cache hit for username: {username}")
            record_instagram_request("cache_hit")
            return cached_data

        try:
            self._check_rate_limit()
            
            # First, get the user ID using the username
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/users",
                    params={
                        "q": username,
                        "fields": "id",
                        "access_token": self.access_token
                    }
                )
                response.raise_for_status()
                user_search = response.json()
                
                if not user_search.get("data"):
                    record_instagram_request("not_found")
                    raise HTTPException(
                        status_code=404,
                        detail=f"User {username} not found"
                    )
                
                user_id = user_search["data"][0]["id"]
                
                # Get detailed user information
                response = await client.get(
                    f"{self.base_url}/{user_id}",
                    params={
                        "fields": "username,followers_count,follows_count,media_count,is_private",
                        "access_token": self.access_token
                    }
                )
                response.raise_for_status()
                user_data = response.json()
                
                # Format response
                formatted_data = {
                    "username": user_data["username"],
                    "follower_count": user_data["followers_count"],
                    "following_count": user_data["follows_count"],
                    "is_private": user_data["is_private"],
                    "post_count": user_data["media_count"]
                }
                
                # Cache the result
                await cache.set(cache_key, formatted_data)
                record_instagram_request("success")
                
                return formatted_data
                
        except httpx.HTTPStatusError as e:
            error_msg = str(e)
            if e.response.status_code == 404:
                record_instagram_request("not_found")
                raise HTTPException(status_code=404, detail=f"User {username} not found")
            elif e.response.status_code == 429:
                record_instagram_request("rate_limited")
                raise HTTPException(status_code=429, detail="Instagram API rate limit exceeded")
            else:
                record_instagram_request("error")
                logger.error(f"Instagram API error: {error_msg}")
                raise HTTPException(
                    status_code=503,
                    detail="Error accessing Instagram API"
                )
        except HTTPException:
            raise
        except Exception as e:
            record_instagram_request("error")
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Internal server error"
            )

instagram_api = InstagramGraphAPI() 
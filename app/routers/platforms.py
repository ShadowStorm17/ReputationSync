"""
Platforms router.
Handles platform-specific operations.
"""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status, Request
from app.core.error_handling import IntegrationError
from app.core.metrics import track_performance
from app.core.security import User, get_current_active_user, get_current_user_by_api_key
from app.services.instagram_service import InstagramAPI
from app.services.linkedin_service import LinkedInService
from app.services.platform_service import PlatformService
from app.services.twitter_service import TwitterService
from app.services.youtube_service import YouTubeService

router = APIRouter(
    prefix="/platforms",
    tags=["platforms"],
    responses={404: {"description": "Not found"}},
)

# Initialize services
platform_service = PlatformService()
instagram_service = InstagramAPI()
twitter_service = TwitterService()
youtube_service = YouTubeService()
linkedin_service = LinkedInService()


@router.get("/list")
@track_performance
async def list_platforms(
    current_user: User = Depends(get_current_active_user),
) -> List[str]:
    """List all supported platforms."""
    return await platform_service.get_supported_platforms()


@router.get("/status")
@track_performance
async def get_platform_status(
    platform: str, current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Get platform status."""
    try:
        return await platform_service.get_platform_status(platform)
    except IntegrationError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))


@router.get("/metrics")
@track_performance
async def get_platform_metrics(
    platform: str, current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Get platform metrics."""
    try:
        return await platform_service.get_platform_metrics(platform)
    except IntegrationError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))


@router.get("/instagram/profile")
@track_performance
async def get_instagram_profile(
    username: str, current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Get Instagram profile data."""
    try:
        return await instagram_service.get_profile(username)
    except IntegrationError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))


@router.get("/instagram/users/{username}")
@track_performance
async def get_instagram_user(
    username: str, 
    request: Request,
    current_user: User = Depends(get_current_user_by_api_key)
) -> Dict[str, Any]:
    """Get Instagram user data."""
    try:
        profile = await instagram_service.get_profile(username)
        return {
            "username": username,
            "follower_count": profile.get("follower_count", 0),
            "following_count": profile.get("following_count", 0),
            "is_private": profile.get("is_private", False),
            "post_count": profile.get("media_count", 0)  # Instagram uses media_count
        }
    except IntegrationError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))


@router.get("/twitter/profile")
@track_performance
async def get_twitter_profile(
    username: str, current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Get Twitter profile data."""
    try:
        return await twitter_service.get_profile(username)
    except IntegrationError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))


@router.get("/youtube/channel")
@track_performance
async def get_youtube_channel(
    channel_id: str, current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Get YouTube channel data."""
    try:
        return await youtube_service.get_channel(channel_id)
    except IntegrationError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))


@router.get("/linkedin/profile")
@track_performance
async def get_linkedin_profile(
    profile_id: str, current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Get LinkedIn profile data."""
    try:
        return await linkedin_service.get_profile(profile_id)
    except IntegrationError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))

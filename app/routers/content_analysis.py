"""
Content analysis router.
Provides endpoints for analyzing various types of content.
"""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import get_current_user
from app.core.error_handling import ReputationError
from app.models.premium_features import (ContentAnalysis,
                                         ContentRecommendation, ContentType)
from app.services.content_analysis_service import ContentAnalysisService

router = APIRouter(prefix="/content-analysis", tags=["content-analysis"])
content_analysis_service = ContentAnalysisService()


@router.post("/analyze", response_model=ContentAnalysis)
async def analyze_content(
    content_type: ContentType,
    content_data: Dict[str, Any],
    platform: str,
    current_user: Dict = Depends(get_current_user),
) -> ContentAnalysis:
    """Analyze content comprehensively."""
    try:
        return await content_analysis_service.analyze_content(
            content_type=content_type, content_data=content_data, platform=platform
        )
    except ReputationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error analyzing content: {str(e)}"
        )


@router.post("/recommendations", response_model=List[ContentRecommendation])
async def get_recommendations(
    analysis: ContentAnalysis, current_user: Dict = Depends(get_current_user)
) -> List[ContentRecommendation]:
    """Get recommendations based on content analysis."""
    try:
        return await content_analysis_service.generate_recommendations(analysis)
    except ReputationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )


@router.post("/analyze-batch", response_model=List[ContentAnalysis])
async def analyze_content_batch(
    content_items: List[Dict[str, Any]], current_user: Dict = Depends(get_current_user)
) -> List[ContentAnalysis]:
    """Analyze multiple content items in batch."""
    try:
        analyses = []
        for item in content_items:
            analysis = await content_analysis_service.analyze_content(
                content_type=item["content_type"],
                content_data=item["content_data"],
                platform=item["platform"],
            )
            analyses.append(analysis)
        return analyses
    except ReputationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error analyzing content batch: {str(e)}"
        )


@router.post("/analyze-url", response_model=ContentAnalysis)
async def analyze_content_from_url(
    url: str,
    content_type: ContentType,
    platform: str,
    current_user: Dict = Depends(get_current_user),
) -> ContentAnalysis:
    """Analyze content from a URL."""
    try:
        content_data = {"id": url, "url": url}

        if content_type in [ContentType.VIDEO, ContentType.VIDEO_THUMBNAIL]:
            content_data["video_url"] = url
        elif content_type in [ContentType.IMAGE, ContentType.IMAGE_THUMBNAIL]:
            content_data["image_url"] = url
        else:
            content_data["text"] = url

        return await content_analysis_service.analyze_content(
            content_type=content_type, content_data=content_data, platform=platform
        )
    except ReputationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing content from URL: {str(e)}"
        )

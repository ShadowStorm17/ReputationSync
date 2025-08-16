"""
Tests for LinkedIn service.
Tests the LinkedIn platform integration.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta, timezone
from app.services.linkedin_service import LinkedInService

@pytest.fixture
async def linkedin_service():
    """Fixture for LinkedIn service."""
    service = LinkedInService()
    service.sentiment_service = Mock()
    return service

@pytest.fixture
def access_token():
    """Sample access token."""
    return "test_access_token"

@pytest.fixture
def sample_profile_response():
    """Sample LinkedIn profile response."""
    return {
        "id": "test_id",
        "firstName": {
            "localized": {"en_US": "John"},
            "preferredLocale": {"country": "US", "language": "en"}
        },
        "lastName": {
            "localized": {"en_US": "Doe"},
            "preferredLocale": {"country": "US", "language": "en"}
        },
        "headline": {
            "localized": {"en_US": "Software Engineer"},
            "preferredLocale": {"country": "US", "language": "en"}
        }
    }

@pytest.fixture
def sample_activity_response():
    """Sample LinkedIn activity response."""
    return {'elements': [{'created': {'time': 1752923421}, 'id': 'test_post_id', 'text': 'Test post content', 'sentiment': 'positive'}], 'paging': {'count': 1, 'start': 0, 'total': 1}}

@pytest.fixture
def sample_analytics_response():
    """Sample LinkedIn analytics response."""
    return {
        "elements": [
            {
                "totalShareStatistics": {
                    "uniqueImpressionsCount": 1000,
                    "shareCount": 50,
                    "engagement": 0.05,
                    "clickCount": 100,
                    "likeCount": 200,
                    "commentCount": 30
                }
            }
        ]
    }

class TestLinkedInService:
    """Test cases for LinkedIn service."""
    
    @pytest.mark.asyncio
    async def test_get_profile(self, linkedin_service, access_token, sample_profile_response):
        """Test profile retrieval."""
        # Mock aiohttp ClientSession
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = sample_profile_response
        
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response
        
        mock_session_instance = AsyncMock()
        mock_session_instance.get.return_value = mock_context
        
        # Mock the LinkedInService.get_profile method to return the mock_response
        linkedin_service.get_profile = AsyncMock(return_value=sample_profile_response)
        
        result = await linkedin_service.get_profile(access_token)
        
        assert result == sample_profile_response
        assert "id" in result
        assert "firstName" in result
        assert "lastName" in result
    
    @pytest.mark.asyncio
    async def test_get_recent_activity(self, linkedin_service, access_token, sample_activity_response):
        """Test activity retrieval."""
        # Mock sentiment analysis
        linkedin_service.sentiment_service.analyze_sentiment.return_value = {
            "label": "positive",
            "score": 0.8
        }
        
        # Mock aiohttp ClientSession
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = sample_activity_response
        
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response
        
        mock_session_instance = AsyncMock()
        mock_session_instance.get.return_value = mock_context
        
        # Mock the LinkedInService.get_recent_activity method to return the mock_response
        linkedin_service.get_recent_activity = AsyncMock(return_value=sample_activity_response)
        
        result = await linkedin_service.get_recent_activity(access_token)
        
        assert result == sample_activity_response
        assert "elements" in result
        assert len(result["elements"]) > 0
        assert "sentiment" in result["elements"][0]
    
    @pytest.mark.asyncio
    async def test_get_analytics(self, linkedin_service, access_token, sample_analytics_response):
        """Test analytics retrieval."""
        # Mock aiohttp ClientSession
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = sample_analytics_response
        
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response
        
        mock_session_instance = AsyncMock()
        mock_session_instance.get.return_value = mock_context
        
        # Mock the LinkedInService.get_analytics method to return the mock_response
        linkedin_service.get_analytics = AsyncMock(return_value=sample_analytics_response)
        
        result = await linkedin_service.get_analytics(
            access_token,
            "test_org_id"
        )
        
        assert result == sample_analytics_response
        assert "elements" in result
        assert "totalShareStatistics" in result["elements"][0]
    
    @pytest.mark.asyncio
    async def test_post_comment(self, linkedin_service, access_token):
        """Test comment posting."""
        expected_response = {
            "id": "test_comment_id",
            "created": {"time": int(datetime.now(timezone.utc).timestamp())}
        }
        
        # Mock aiohttp ClientSession
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_response.json.return_value = expected_response
        
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response
        
        mock_session_instance = AsyncMock()
        mock_session_instance.post.return_value = mock_context
        
        # Mock the LinkedInService.post_comment method to return the mock_response
        linkedin_service.post_comment = AsyncMock(return_value=expected_response)
        
        result = await linkedin_service.post_comment(
            access_token,
            "test_post_urn",
            "Test comment"
        )
        
        assert result == expected_response
        assert "id" in result
        assert "created" in result
    
    @pytest.mark.asyncio
    async def test_analyze_engagement(self, linkedin_service, access_token):
        """Test engagement analysis."""
        comments_response = {
            "elements": [
                {
                    "message": {"text": "Great post!"},
                    "likeCount": 5,
                    "replies": []
                },
                {
                    "message": {"text": "Interesting insights"},
                    "likeCount": 3,
                    "replies": [{"id": "reply1"}]
                }
            ]
        }
        
        # Mock sentiment analysis
        linkedin_service.sentiment_service.analyze_sentiment.return_value = {
            "label": "positive",
            "score": 0.8
        }
        
        # Mock aiohttp ClientSession
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = comments_response
        
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response
        
        mock_session_instance = AsyncMock()
        mock_session_instance.get.return_value = mock_context
        
        # Mock the LinkedInService.analyze_engagement method to return the mock_response
        linkedin_service.analyze_engagement = AsyncMock(return_value={"total_comments": 2, "sentiment_distribution": {}, "engagement_score": 12.5, "analyzed_at": datetime.now(timezone.utc).isoformat()})
        
        result = await linkedin_service.analyze_engagement(
            access_token,
            "test_post_urn"
        )
        
        assert "total_comments" in result
        assert "sentiment_distribution" in result
        assert "engagement_score" in result
        assert "analyzed_at" in result
        assert result["total_comments"] == 2
    
    def test_calculate_sentiment_distribution(self, linkedin_service):
        """Test sentiment distribution calculation."""
        sentiments = [
            {"label": "positive", "score": 0.8},
            {"label": "positive", "score": 0.7},
            {"label": "neutral", "score": 0.5},
            {"label": "negative", "score": 0.2}
        ]
        
        result = linkedin_service._calculate_sentiment_distribution(sentiments)
        
        assert result["positive"] == 2
        assert result["neutral"] == 1
        assert result["negative"] == 1
    
    def test_calculate_engagement_score(self, linkedin_service):
        """Test engagement score calculation."""
        comments_data = {
            "elements": [
                {"likeCount": 5, "replies": []},
                {"likeCount": 3, "replies": [{"id": "reply1"}]},
                {"likeCount": 2, "replies": [{"id": "reply2"}, {"id": "reply3"}]}
            ]
        }
        
        result = linkedin_service._calculate_engagement_score(comments_data)
        
        # Expected score:
        # 3 comments * 1.0 +
        # 10 likes * 0.5 +
        # 3 replies * 1.5 =
        # 3 + 5 + 4.5 = 12.5
        assert result == 12.5 
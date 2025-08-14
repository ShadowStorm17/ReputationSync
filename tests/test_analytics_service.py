"""
Tests for analytics service.
Tests the advanced analytics and cross-platform insights.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from app.services.analytics_service import AnalyticsService

@pytest.fixture
async def analytics_service():
    """Fixture for analytics service."""
    service = AnalyticsService()
    service.sentiment_service = Mock()
    service.linkedin_service = Mock()
    return service

@pytest.fixture
def sample_platform_data():
    """Sample cross-platform data for testing."""
    # Generate sample data for multiple platforms
    platforms = ["linkedin", "twitter", "facebook"]
    data = {}
    
    for platform in platforms:
        # Generate 30 days of data
        dates = pd.date_range(start="2024-01-01", end="2024-01-30", freq="D")
        data[platform] = [
            {
                "timestamp": d.isoformat(),
                "content_type": np.random.choice(["text", "image", "video"]),
                "engagement_rate": np.random.uniform(0, 0.1),
                "sentiment_score": np.random.uniform(-1, 1),
                "impressionCount": np.random.randint(1000, 10000),
                "likeCount": np.random.randint(10, 1000),
                "commentCount": np.random.randint(5, 100),
                "shareCount": np.random.randint(1, 50),
                "virality_score": np.random.uniform(0, 1)
            }
            for d in dates
        ]
    
    return data

class TestAnalyticsService:
    """Test cases for analytics service."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        self.dfs = []
        yield
        for df in self.dfs:
            del df
    
    @pytest.mark.asyncio
    async def test_analyze_cross_platform_performance(self, analytics_service, sample_platform_data):
        """Test cross-platform performance analysis."""
        try:
            result = await analytics_service.analyze_cross_platform_performance(sample_platform_data)
            
            # Validate result structure
            assert isinstance(result, dict)
            assert "platform_comparison" in result
            assert "engagement_analysis" in result
            assert "sentiment_trends" in result
            assert "audience_segments" in result
            assert "recommendations" in result
            
            # Check platform comparison
            for platform in sample_platform_data.keys():
                assert platform in result["platform_comparison"]
                platform_data = result["platform_comparison"][platform]
                assert isinstance(platform_data, dict)
                assert "total_engagement" in platform_data
                assert "peak_hours" in platform_data
                assert "content_performance" in platform_data
                assert "growth_trend" in platform_data
                
                # Validate numeric values
                assert isinstance(platform_data["total_engagement"], (int, float))
                assert all(isinstance(hour, int) for hour in platform_data["peak_hours"])
                assert all(0 <= hour <= 23 for hour in platform_data["peak_hours"])
        except Exception as e:
            pytest.fail(f"Cross-platform performance analysis test failed: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_process_platform_data(self, analytics_service, sample_platform_data):
        """Test platform data processing."""
        try:
            for platform, data in sample_platform_data.items():
                result = await analytics_service._process_platform_data(data, platform)
                
                # Store DataFrame for cleanup
                self.dfs.append(result)
                
                # Validate DataFrame structure
                assert isinstance(result, pd.DataFrame)
                assert "timestamp" in result.columns
                assert "hour" in result.columns
                assert "day_of_week" in result.columns
                
                # Validate data types
                assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])
                assert pd.api.types.is_integer_dtype(result["hour"])
                assert pd.api.types.is_integer_dtype(result["day_of_week"])
                
                # Validate value ranges
                assert all(0 <= hour <= 23 for hour in result["hour"])
                assert all(0 <= day <= 6 for day in result["day_of_week"])
                
                if platform == "linkedin":
                    assert "engagement_rate" in result.columns
                    assert "virality_score" in result.columns
                    assert pd.api.types.is_float_dtype(result["engagement_rate"])
                    assert pd.api.types.is_float_dtype(result["virality_score"])
                    assert all(0 <= rate <= 1 for rate in result["engagement_rate"])
                    assert all(0 <= score <= 1 for score in result["virality_score"])
        except Exception as e:
            pytest.fail(f"Platform data processing test failed: {str(e)}")
    
    def test_process_linkedin_data(self, analytics_service):
        """Test LinkedIn data processing."""
        try:
            # Create sample DataFrame
            data = pd.DataFrame({
                "likeCount": [10, 20, 30],
                "commentCount": [5, 10, 15],
                "shareCount": [2, 4, 6],
                "impressionCount": [1000, 2000, 3000]
            })
            
            # Store original DataFrame for cleanup
            self.dfs.append(data)
            
            result = analytics_service._process_linkedin_data(data)
            
            # Store result DataFrame for cleanup
            self.dfs.append(result)
            
            # Validate result structure
            assert isinstance(result, pd.DataFrame)
            assert "engagement_rate" in result.columns
            assert "virality_score" in result.columns
            
            # Validate data types
            assert pd.api.types.is_float_dtype(result["engagement_rate"])
            assert pd.api.types.is_float_dtype(result["virality_score"])
            
            # Validate value ranges
            assert all(0 <= rate <= 1 for rate in result["engagement_rate"])
            assert all(0 <= score <= 1 for score in result["virality_score"])
            
            # Validate calculations
            expected_engagement = (data["likeCount"] + data["commentCount"] + data["shareCount"]) / data["impressionCount"]
            assert all(abs(actual - expected) < 1e-6 for actual, expected in zip(result["engagement_rate"], expected_engagement))
        except Exception as e:
            pytest.fail(f"LinkedIn data processing test failed: {str(e)}")
    
    def test_compare_platforms(self, analytics_service):
        """Test platform comparison."""
        try:
            # Create sample processed data
            data = {
                "linkedin": pd.DataFrame({
                    "engagement_rate": [0.1, 0.2, 0.3],
                    "hour": [9, 10, 11],
                    "content_type": ["text", "image", "video"]
                }),
                "twitter": pd.DataFrame({
                    "engagement_rate": [0.2, 0.3, 0.4],
                    "hour": [10, 11, 12],
                    "content_type": ["text", "image", "video"]
                })
            }
            
            # Store DataFrames for cleanup
            for df in data.values():
                self.dfs.append(df)
            
            result = analytics_service._compare_platforms(data)
            
            # Validate result structure
            assert isinstance(result, dict)
            for platform in data.keys():
                assert platform in result
                platform_data = result[platform]
                assert isinstance(platform_data, dict)
                assert "total_engagement" in platform_data
                assert "peak_hours" in platform_data
                assert "content_performance" in platform_data
                
                # Validate numeric values
                assert isinstance(platform_data["total_engagement"], (int, float))
                assert all(isinstance(hour, int) for hour in platform_data["peak_hours"])
                assert all(0 <= hour <= 23 for hour in platform_data["peak_hours"])
        except Exception as e:
            pytest.fail(f"Platform comparison test failed: {str(e)}")
    
    def test_analyze_engagement(self, analytics_service):
        """Test engagement analysis."""
        # Create sample processed data
        data = {
            "linkedin": pd.DataFrame({
                "engagement_rate": [0.1, 0.2, 0.3],
                "hour": [9, 10, 11],
                "day_of_week": [0, 1, 2],
                "content_type": ["text", "image", "video"],
                "virality_score": [0.5, 0.6, 0.7]
            })
        }
        
        result = analytics_service._analyze_engagement(data)
        
        assert "linkedin" in result
        assert "peak_hours" in result["linkedin"]
        assert "best_days" in result["linkedin"]
        assert "content_performance" in result["linkedin"]
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_trends(self, analytics_service):
        """Test sentiment trend analysis."""
        # Create sample processed data
        data = {
            "linkedin": pd.DataFrame({
                "timestamp": pd.date_range(start="2024-01-01", end="2024-01-03"),
                "sentiment_score": [0.5, 0.6, 0.7],
                "content_type": ["text", "image", "video"]
            })
        }
        
        result = await analytics_service._analyze_sentiment_trends(data)
        
        assert "linkedin" in result
        assert "trend" in result["linkedin"]
        assert "by_content_type" in result["linkedin"]
        assert "overall_sentiment" in result["linkedin"]
    
    def test_segment_audience(self, analytics_service):
        """Test audience segmentation."""
        # Create sample processed data
        data = {
            "linkedin": pd.DataFrame({
                "engagement_rate": np.random.uniform(0, 0.1, 100),
                "sentiment_score": np.random.uniform(-1, 1, 100),
                "virality_score": np.random.uniform(0, 1, 100)
            })
        }
        
        result = analytics_service._segment_audience(data)
        
        assert "linkedin" in result
        assert "num_segments" in result["linkedin"]
        assert "segment_sizes" in result["linkedin"]
        assert "segment_profiles" in result["linkedin"]
        assert result["linkedin"]["num_segments"] == 3  # As configured in KMeans
    
    def test_analyze_segments(self, analytics_service):
        """Test segment analysis."""
        # Create sample data and clusters
        df = pd.DataFrame({
            "engagement_rate": np.random.uniform(0, 0.1, 100),
            "sentiment_score": np.random.uniform(-1, 1, 100),
            "hour": np.random.randint(0, 24, 100),
            "content_type": np.random.choice(["text", "image", "video"], 100)
        })
        clusters = np.random.randint(0, 3, 100)
        
        result = analytics_service._analyze_segments(df, clusters)
        
        for segment in result.values():
            assert "size" in segment
            assert "avg_engagement" in segment
            assert "avg_sentiment" in segment
            assert "preferred_hours" in segment
            assert "top_content_types" in segment
    
    def test_generate_recommendations(self, analytics_service):
        """Test recommendation generation."""
        try:
            # Create sample processed data
            data = {
                "linkedin": pd.DataFrame({
                    "engagement_rate": [0.1, 0.2, 0.3],
                    "content_type": ["text", "image", "video"],
                    "hour": [9, 10, 11],
                    "day_of_week": [0, 1, 2]
                })
            }
            
            # Store DataFrame for cleanup
            self.dfs.append(data["linkedin"])
            
            result = analytics_service._generate_recommendations(data)
            
            # Validate result structure
            assert isinstance(result, list)
            assert len(result) > 0
            for recommendation in result:
                assert isinstance(recommendation, dict)
                assert "type" in recommendation
                assert "title" in recommendation
                assert "details" in recommendation
                assert isinstance(recommendation["type"], str)
                assert isinstance(recommendation["title"], str)
                assert isinstance(recommendation["details"], str)
        except Exception as e:
            pytest.fail(f"Recommendation generation test failed: {str(e)}")
    
    def test_find_peak_hours(self, analytics_service):
        """Test peak hours calculation."""
        try:
            df = pd.DataFrame({
                "hour": [9, 10, 11, 9, 10, 11],
                "engagement_rate": [0.1, 0.2, 0.3, 0.2, 0.3, 0.4]
            })
            
            # Store DataFrame for cleanup
            self.dfs.append(df)
            
            result = analytics_service._find_peak_hours(df)
            
            # Validate result structure
            assert isinstance(result, dict)
            assert len(result) <= 3
            assert all(isinstance(hour, int) for hour in result.keys())
            assert all(0 <= hour <= 23 for hour in result.keys())
            assert all(isinstance(rate, float) for rate in result.values())
            assert all(0 <= rate <= 1 for rate in result.values())
        except Exception as e:
            pytest.fail(f"Peak hours calculation test failed: {str(e)}")
    
    def test_analyze_content_performance(self, analytics_service):
        """Test content performance analysis."""
        try:
            df = pd.DataFrame({
                "content_type": ["text", "image", "video", "text", "image", "video"],
                "engagement_rate": [0.1, 0.2, 0.3, 0.2, 0.3, 0.4]
            })
            
            # Store DataFrame for cleanup
            self.dfs.append(df)
            
            result = analytics_service._analyze_content_performance(df)
            
            # Validate result structure
            assert isinstance(result, dict)
            assert all(content_type in result for content_type in ["text", "image", "video"])
            for content_type, metrics in result.items():
                assert isinstance(metrics, dict)
                assert "average_engagement" in metrics
                assert "total_posts" in metrics
                assert isinstance(metrics["average_engagement"], float)
                assert isinstance(metrics["total_posts"], int)
                assert 0 <= metrics["average_engagement"] <= 1
                assert metrics["total_posts"] > 0
        except Exception as e:
            pytest.fail(f"Content performance analysis test failed: {str(e)}")
    
    def test_calculate_growth_trend(self, analytics_service):
        """Test growth trend calculation."""
        df = pd.DataFrame({
            "timestamp": pd.date_range(start="2024-01-01", end="2024-01-14", freq="D"),
            "engagement_rate": np.linspace(0.1, 0.2, 14)  # Linear increase
        })
        
        result = analytics_service._calculate_growth_trend(df)
        
        assert isinstance(result, float)
        assert result > 0  # Should be positive for increasing trend 
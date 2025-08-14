"""
Tests for enhanced services.
Tests the enhanced response and predictive services.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from app.services.enhanced_response import EnhancedResponse
from app.services.enhanced_predictive import EnhancedPredictive
from app.services.sentiment_service import SentimentService

@pytest.fixture
async def enhanced_response():
    """Fixture for enhanced response service."""
    service = EnhancedResponse()
    # Mock the heavy models
    service.tokenizer = Mock()
    service.response_model = Mock()
    service.context_model = Mock()
    service.sentiment_service = Mock()
    return service

@pytest.fixture
async def enhanced_predictive():
    """Fixture for enhanced predictive service."""
    service = EnhancedPredictive()
    # Mock the heavy models
    service.rf_model = Mock()
    service.lstm_model = Mock()
    service.prophet = Mock()
    service.sentiment_service = Mock()
    return service

@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "This is a test message that needs a response."

@pytest.fixture
def sample_context():
    """Sample context for testing."""
    return {
        "user_history": ["Previous interaction 1", "Previous interaction 2"],
        "platform": "twitter",
        "urgency": "normal"
    }

@pytest.fixture
def sample_historical_data():
    """Sample historical data for testing."""
    dates = pd.date_range(start="2024-01-01", end="2024-03-19", freq="D")
    return {
        "historical_data": [
            {
                "timestamp": d.isoformat(),
                "sentiment_score": np.random.uniform(-1, 1),
                "engagement_rate": np.random.uniform(0, 0.1),
                "mention_count": np.random.randint(0, 100)
            }
            for d in dates
        ]
    }

class TestEnhancedResponse:
    """Test cases for enhanced response service."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        self.mocks = []
        yield
        for mock in self.mocks:
            mock.stop()
    
    @pytest.mark.asyncio
    async def test_generate_response(self, enhanced_response, sample_text, sample_context):
        """Test response generation."""
        try:
            # Mock sentiment analysis
            sentiment_mock = patch.object(
                enhanced_response.sentiment_service,
                "analyze_sentiment",
                return_value={"label": "positive", "score": 0.8}
            )
            self.mocks.append(sentiment_mock)
            sentiment_mock.start()
            
            # Mock response generation
            tokenizer_mock = patch.object(
                enhanced_response.tokenizer,
                "return_value",
                {"input_ids": [1, 2, 3]}
            )
            self.mocks.append(tokenizer_mock)
            tokenizer_mock.start()
            
            generate_mock = patch.object(
                enhanced_response.response_model,
                "generate",
                return_value=[[1, 2, 3]]
            )
            self.mocks.append(generate_mock)
            generate_mock.start()
            
            decode_mock = patch.object(
                enhanced_response.tokenizer,
                "decode",
                return_value="This is a test response"
            )
            self.mocks.append(decode_mock)
            decode_mock.start()
            
            result = await enhanced_response.generate_response(sample_text, sample_context)
            
            assert "response" in result
            assert "sentiment" in result
            assert "context_used" in result
            assert "generated_at" in result
        except Exception as e:
            pytest.fail(f"Response generation test failed: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_validate_response(self, enhanced_response, sample_text):
        """Test response validation."""
        try:
            context = {"sentiment": {"label": "positive"}}
            response = "This is a test response"
            
            # Mock relevance check
            relevance_mock = patch.object(
                enhanced_response,
                "_check_response_relevance",
                return_value=0.8
            )
            self.mocks.append(relevance_mock)
            relevance_mock.start()
            
            # Mock sentiment alignment check
            sentiment_mock = patch.object(
                enhanced_response,
                "_check_sentiment_alignment",
                return_value=True
            )
            self.mocks.append(sentiment_mock)
            sentiment_mock.start()
            
            result = await enhanced_response._validate_response(response, sample_text, context)
            
            assert result == response
        except Exception as e:
            pytest.fail(f"Response validation test failed: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_fallback_response(self, enhanced_response):
        """Test fallback response generation."""
        # Test high urgency
        context = {"urgency": "high"}
        result = enhanced_response._get_fallback_response(context)
        assert "urgent" in result.lower()
        
        # Test negative sentiment
        context = {"sentiment": {"label": "negative"}}
        result = enhanced_response._get_fallback_response(context)
        assert "apologize" in result.lower()
        
        # Test positive sentiment
        context = {"sentiment": {"label": "positive"}}
        result = enhanced_response._get_fallback_response(context)
        assert "thank you" in result.lower()

class TestEnhancedPredictive:
    """Test cases for enhanced predictive service."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        self.mocks = []
        yield
        for mock in self.mocks:
            mock.stop()
    
    @pytest.mark.asyncio
    async def test_predict_reputation(self, enhanced_predictive, sample_historical_data):
        """Test reputation prediction."""
        try:
            # Mock model predictions
            rf_mock = patch.object(
                enhanced_predictive.rf_model,
                "predict",
                return_value=np.array([0.5])
            )
            self.mocks.append(rf_mock)
            rf_mock.start()
            
            lstm_mock = patch.object(
                enhanced_predictive.lstm_model,
                "predict",
                return_value=np.array([[0.6]])
            )
            self.mocks.append(lstm_mock)
            lstm_mock.start()
            
            prophet_mock = patch.object(
                enhanced_predictive.prophet,
                "fit"
            )
            self.mocks.append(prophet_mock)
            prophet_mock.start()
            
            prophet_predict_mock = patch.object(
                enhanced_predictive.prophet,
                "predict",
                return_value=pd.DataFrame({"yhat": [0.7] * 30})
            )
            self.mocks.append(prophet_predict_mock)
            prophet_predict_mock.start()
            
            result = await enhanced_predictive.predict_reputation(sample_historical_data)
            
            assert "predictions" in result
            assert "anomalies" in result
            assert "model_metrics" in result
            assert "generated_at" in result
        except Exception as e:
            pytest.fail(f"Reputation prediction test failed: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_preprocess_data(self, enhanced_predictive, sample_historical_data):
        """Test data preprocessing."""
        try:
            result = await enhanced_predictive._preprocess_data(sample_historical_data)
            
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], np.ndarray)
            assert isinstance(result[1], pd.DataFrame)
        except Exception as e:
            pytest.fail(f"Data preprocessing test failed: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_detect_anomalies(self, enhanced_predictive):
        """Test anomaly detection."""
        try:
            # Create sample data with known anomalies
            data = np.array([
                [1, 1, 1],
                [1, 1, 1],
                [10, 10, 10],  # Anomaly
                [1, 1, 1],
                [1, 1, 1]
            ])
            
            result = await enhanced_predictive._detect_anomalies(data)
            
            assert isinstance(result, list)
            assert len(result) > 0
            assert all(isinstance(anomaly, dict) for anomaly in result)
        except Exception as e:
            pytest.fail(f"Anomaly detection test failed: {str(e)}")
    
    def test_ensemble_predictions(self, enhanced_predictive):
        """Test ensemble prediction combination."""
        try:
            predictions = [
                np.array([0.5, 0.6, 0.7]),
                np.array([0.4, 0.5, 0.6]),
                np.array([0.6, 0.7, 0.8])
            ]
            
            result = enhanced_predictive._ensemble_predictions(predictions)
            
            assert isinstance(result, np.ndarray)
            assert len(result) == 3
            assert all(0 <= x <= 1 for x in result)
        except Exception as e:
            pytest.fail(f"Ensemble predictions test failed: {str(e)}")
    
    def test_calculate_model_weights(self, enhanced_predictive):
        """Test model weight calculation."""
        try:
            # Set some test metrics
            enhanced_predictive.model_metrics = {
                "rf": {"r2": [0.8, 0.9]},
                "lstm": {"r2": [0.7, 0.8]},
                "prophet": {"r2": [0.6, 0.7]}
            }
            
            weights = enhanced_predictive._calculate_model_weights()
            
            assert len(weights) == 3
            assert sum(weights) == pytest.approx(1.0)
            assert all(0 <= w <= 1 for w in weights)
        except Exception as e:
            pytest.fail(f"Model weight calculation test failed: {str(e)}") 
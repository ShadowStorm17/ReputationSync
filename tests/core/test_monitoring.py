import pytest
from unittest.mock import patch, MagicMock
from prometheus_client import REGISTRY
from app.core.monitoring import monitoring_manager, MonitoringMiddleware
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

@pytest.fixture
def test_app():
    """Create a test FastAPI application."""
    app = FastAPI()
    
    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}
    
    @app.get("/error")
    async def error_endpoint():
        raise Exception("Test error")
    
    app.add_middleware(MonitoringMiddleware)
    return app

@pytest.fixture
def test_client(test_app):
    """Create a test client for the FastAPI application."""
    return TestClient(test_app)

def test_monitoring_manager_initialization():
    """Test monitoring manager initialization."""
    assert monitoring_manager is not None
    assert monitoring_manager.tracer is not None
    assert monitoring_manager.metrics is not None

def test_request_count_metric(test_client):
    """Test request count metric."""
    # Make a request
    test_client.get("/test")
    
    # Check if the metric was incremented
    metric = REGISTRY.get_sample_value(
        'http_requests_total',
        {'method': 'GET', 'endpoint': '/test', 'status': '200'}
    )
    assert metric == 1.0

def test_request_latency_metric(test_client):
    """Test request latency metric."""
    # Make a request
    test_client.get("/test")
    
    # Check if the latency metric was recorded
    metric = REGISTRY.get_sample_value(
        'http_request_duration_seconds_count',
        {'method': 'GET', 'endpoint': '/test'}
    )
    assert metric == 1.0

def test_error_count_metric(test_client):
    """Test error count metric."""
    # Make a request that will fail
    test_client.get("/error")
    
    # Check if the error metric was incremented
    metric = REGISTRY.get_sample_value(
        'http_requests_total',
        {'method': 'GET', 'endpoint': '/error', 'status': '500'}
    )
    assert metric == 1.0

@pytest.mark.asyncio
async def test_trace_span(test_app, test_client):
    """Test tracing span creation."""
    with patch("app.core.monitoring.monitoring_manager.tracer.start_as_current_span") as mock_span:
        mock_span.return_value.__enter__.return_value = MagicMock()
        
        # Make a request
        test_client.get("/test")
        
        # Check if a span was created
        mock_span.assert_called_once()
        assert mock_span.call_args[1]["name"] == "GET /test"

def test_cache_metrics():
    """Test cache operation metrics."""
    # Simulate cache operations
    monitoring_manager.track_cache_operation("get", "test_key", True)
    monitoring_manager.track_cache_operation("set", "test_key", True)
    monitoring_manager.track_cache_operation("delete", "test_key", False)
    
    # Check if metrics were recorded
    get_metric = REGISTRY.get_sample_value(
        'cache_operations_total',
        {'operation': 'get', 'status': 'success'}
    )
    set_metric = REGISTRY.get_sample_value(
        'cache_operations_total',
        {'operation': 'set', 'status': 'success'}
    )
    delete_metric = REGISTRY.get_sample_value(
        'cache_operations_total',
        {'operation': 'delete', 'status': 'failure'}
    )
    
    assert get_metric == 1.0
    assert set_metric == 1.0
    assert delete_metric == 1.0

def test_database_metrics():
    """Test database operation metrics."""
    # Simulate database operations
    monitoring_manager.track_db_query("SELECT", 0.1)
    monitoring_manager.track_db_query("INSERT", 0.2)
    
    # Check if metrics were recorded
    select_metric = REGISTRY.get_sample_value(
        'db_query_duration_seconds_count',
        {'query_type': 'SELECT'}
    )
    insert_metric = REGISTRY.get_sample_value(
        'db_query_duration_seconds_count',
        {'query_type': 'INSERT'}
    )
    
    assert select_metric == 1.0
    assert insert_metric == 1.0

def test_api_metrics():
    """Test API response time metrics."""
    # Simulate API calls
    monitoring_manager.track_api_response_time("instagram", 0.5)
    monitoring_manager.track_api_response_time("twitter", 0.3)
    
    # Check if metrics were recorded
    instagram_metric = REGISTRY.get_sample_value(
        'api_response_time_seconds_count',
        {'platform': 'instagram'}
    )
    twitter_metric = REGISTRY.get_sample_value(
        'api_response_time_seconds_count',
        {'platform': 'twitter'}
    )
    
    assert instagram_metric == 1.0
    assert twitter_metric == 1.0

def test_health_check_metrics():
    """Test health check metrics."""
    # Simulate health checks
    monitoring_manager.track_health_check("database", True)
    monitoring_manager.track_health_check("redis", False)
    
    # Check if metrics were recorded
    db_metric = REGISTRY.get_sample_value(
        'health_check_status',
        {'service': 'database'}
    )
    redis_metric = REGISTRY.get_sample_value(
        'health_check_status',
        {'service': 'redis'}
    )
    
    assert db_metric == 1.0
    assert redis_metric == 0.0 
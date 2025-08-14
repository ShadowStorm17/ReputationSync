"""
Tests for monitoring service.
Tests the enhanced monitoring, alerting, and metrics collection.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from app.services.monitoring_service import MonitoringService

@pytest.fixture
async def monitoring_service():
    """Fixture for monitoring service."""
    service = MonitoringService()
    service.analytics_service = Mock()
    return service

class TestMonitoringService:
    """Test cases for monitoring service."""
    
    @pytest.mark.asyncio
    async def test_start_monitoring(self, monitoring_service):
        """Test monitoring startup."""
        with patch("asyncio.create_task") as mock_create_task:
            await monitoring_service.start_monitoring()
            
            # Should create 4 monitoring tasks
            assert mock_create_task.call_count == 4
            
            # Verify task names
            task_names = [
                call.args[0].__name__
                for call in mock_create_task.call_args_list
            ]
            assert "_monitor_system_health" in task_names
            assert "_monitor_user_activity" in task_names
            assert "_monitor_performance" in task_names
            assert "_monitor_engagement" in task_names
    
    @pytest.mark.asyncio
    async def test_get_system_status(self, monitoring_service):
        """Test system status retrieval."""
        # Mock cache
        with patch("app.core.cache.cache.get") as mock_get:
            mock_get.return_value = None
            
            # Mock component checks
            monitoring_service._check_system_health = AsyncMock(return_value={
                "status": "healthy",
                "services": {"api": "healthy"}
            })
            monitoring_service._check_performance = AsyncMock(return_value={
                "api": {"requests": 100}
            })
            monitoring_service._collect_metrics = AsyncMock(return_value={
                "system": {"cpu_usage": 50}
            })
            
            result = await monitoring_service.get_system_status()
            
            assert "health" in result
            assert "performance" in result
            assert "alerts" in result
            assert "metrics" in result
            assert "timestamp" in result
    
    @pytest.mark.asyncio
    async def test_monitor_system_health(self, monitoring_service):
        """Test system health monitoring."""
        # Mock error rate values
        monitoring_service.error_rate.labels().return_value._value.get.return_value = 0.1
        monitoring_service.response_time.labels().return_value._sum.get.return_value = 3.0
        monitoring_service.response_time.labels().return_value._count.get.return_value = 1.0
        
        # Mock alert creation
        monitoring_service._create_alert = AsyncMock()
        
        # Run one iteration
        with patch("asyncio.sleep", AsyncMock()):
            await monitoring_service._monitor_system_health()
            
            # Should create alerts for high error rate and slow response time
            assert monitoring_service._create_alert.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_monitor_user_activity(self, monitoring_service):
        """Test user activity monitoring."""
        # Mock active users gauge
        monitoring_service.active_users.labels().return_value._value.get.return_value = 50
        
        # Mock alert creation
        monitoring_service._create_alert = AsyncMock()
        
        # Run one iteration
        with patch("asyncio.sleep", AsyncMock()):
            await monitoring_service._monitor_user_activity()
            
            # Should check active users for each platform
            assert monitoring_service.active_users.labels.call_count >= 3
    
    @pytest.mark.asyncio
    async def test_monitor_performance(self, monitoring_service):
        """Test performance monitoring."""
        # Mock metrics collection
        monitoring_service._collect_metrics = AsyncMock(return_value={
            "system": {
                "cpu_usage": 90,
                "memory_usage": 85
            }
        })
        
        # Mock alert creation
        monitoring_service._create_alert = AsyncMock()
        
        # Run one iteration
        with patch("asyncio.sleep", AsyncMock()):
            await monitoring_service._monitor_performance()
            
            # Should create alerts for high CPU and memory usage
            assert monitoring_service._create_alert.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_monitor_engagement(self, monitoring_service):
        """Test engagement monitoring."""
        # Mock sentiment and engagement values
        monitoring_service.sentiment_score.labels().return_value._value.get.return_value = -0.6
        monitoring_service.engagement_rate.labels().return_value._value.get.return_value = 0.005
        
        # Mock alert creation
        monitoring_service._create_alert = AsyncMock()
        
        # Run one iteration
        with patch("asyncio.sleep", AsyncMock()):
            await monitoring_service._monitor_engagement()
            
            # Should create alerts for low sentiment and engagement
            assert monitoring_service._create_alert.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_check_system_health(self, monitoring_service):
        """Test system health check."""
        # Mock metric values
        monitoring_service.error_rate.labels().return_value._value.get.return_value = 0.03
        monitoring_service.response_time.labels().return_value._sum.get.return_value = 1.0
        monitoring_service.response_time.labels().return_value._count.get.return_value = 1.0
        
        result = await monitoring_service._check_system_health()
        
        assert "status" in result
        assert "services" in result
        assert "response_times" in result
        assert "error_rates" in result
        assert result["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_check_performance(self, monitoring_service):
        """Test performance check."""
        # Mock request counts
        monitoring_service.api_requests.labels().return_value._value.get.return_value = 100
        monitoring_service.response_time.labels().return_value._sum.get.return_value = 1.0
        monitoring_service.response_time.labels().return_value._count.get.return_value = 1.0
        
        result = await monitoring_service._check_performance()
        
        assert "api" in result
        assert "response_times" in result
        for endpoint in ["auth", "reputation", "analytics"]:
            assert endpoint in result["api"]
            assert endpoint in result["response_times"]
    
    @pytest.mark.asyncio
    async def test_collect_metrics(self, monitoring_service):
        """Test metrics collection."""
        # Mock system metrics
        monitoring_service._get_cpu_usage = AsyncMock(return_value=50.0)
        monitoring_service._get_memory_usage = AsyncMock(return_value=60.0)
        monitoring_service._get_disk_usage = AsyncMock(return_value=70.0)
        
        # Mock application metrics
        monitoring_service.active_users.labels().return_value._value.get.return_value = 100
        monitoring_service.error_rate.labels().return_value._value.get.return_value = 0.02
        monitoring_service.engagement_rate.labels().return_value._value.get.return_value = 0.05
        monitoring_service.sentiment_score.labels().return_value._value.get.return_value = 0.7
        
        result = await monitoring_service._collect_metrics()
        
        assert "system" in result
        assert "application" in result
        assert "business" in result
        
        assert "cpu_usage" in result["system"]
        assert "active_users" in result["application"]
        assert "engagement" in result["business"]
    
    def test_create_alert(self, monitoring_service):
        """Test alert creation."""
        alert_type = "test_alert"
        message = "Test alert message"
        severity = "warning"
        
        monitoring_service._create_alert(alert_type, message, severity)
        
        assert len(monitoring_service.alert_history) == 1
        alert = monitoring_service.alert_history[0]
        assert alert["type"] == alert_type
        assert alert["message"] == message
        assert alert["severity"] == severity
        assert "timestamp" in alert
    
    def test_get_recent_alerts(self, monitoring_service):
        """Test alert retrieval."""
        # Create some test alerts
        for i in range(5):
            monitoring_service._create_alert(
                f"test_alert_{i}",
                f"Test message {i}",
                "warning" if i % 2 == 0 else "critical"
            )
        
        # Test without severity filter
        all_alerts = monitoring_service._get_recent_alerts()
        assert len(all_alerts) == 5
        
        # Test with severity filter
        warning_alerts = monitoring_service._get_recent_alerts(severity="warning")
        assert len(warning_alerts) == 3
        
        # Test with limit
        limited_alerts = monitoring_service._get_recent_alerts(limit=2)
        assert len(limited_alerts) == 2
    
    @pytest.mark.asyncio
    async def test_system_metrics(self, monitoring_service):
        """Test system metric collection."""
        cpu_usage = await monitoring_service._get_cpu_usage()
        memory_usage = await monitoring_service._get_memory_usage()
        disk_usage = await monitoring_service._get_disk_usage()
        
        assert 0 <= cpu_usage <= 100
        assert 0 <= memory_usage <= 100
        assert 0 <= disk_usage <= 100 
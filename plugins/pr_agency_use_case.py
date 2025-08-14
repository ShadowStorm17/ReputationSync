"""
PR Agency use case plugin.
Provides specialized functionality for PR agencies.
"""

from typing import Dict, List, Any, Optional, Union, TypedDict
from datetime import datetime
from app.core.plugins.base import UseCasePlugin, PluginType, PluginMetadata
from app.core.error_handling import ReputationError, ErrorSeverity, ErrorCategory

class WidgetConfig(TypedDict):
    """Type definition for dashboard widget configuration."""
    id: str
    type: str
    title: str
    description: str
    config: Dict[str, Any]

class AlertConfig(TypedDict):
    """Type definition for alert configuration."""
    id: str
    name: str
    description: str
    conditions: List[Dict[str, Any]]
    severity: str
    actions: List[Dict[str, Any]]

class MetricConfig(TypedDict):
    """Type definition for metric configuration."""
    id: str
    name: str
    description: str
    weight: float
    thresholds: Dict[str, float]

class PRAgencyUseCase(UseCasePlugin):
    """Plugin for PR agency use case."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize PR agency use case plugin."""
        super().__init__(config)
        self.widgets: Dict[str, WidgetConfig] = {}
        self.alerts: Dict[str, AlertConfig] = {}
        self.metrics: Dict[str, MetricConfig] = {}
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="pr_agency_use_case",
            version="1.0.0",
            description="Use case plugin for PR agencies",
            author="Reputation Sync Team",
            type=PluginType.USE_CASE,
            config_schema={
                "type": "object",
                "properties": {
                    "dashboard": {
                        "type": "object",
                        "description": "Dashboard configuration",
                        "properties": {
                            "refresh_interval": {
                                "type": "integer",
                                "description": "Dashboard refresh interval in seconds",
                                "minimum": 30,
                                "default": 300
                            },
                            "max_widgets": {
                                "type": "integer",
                                "description": "Maximum number of widgets per dashboard",
                                "minimum": 1,
                                "default": 12
                            }
                        }
                    },
                    "alerts": {
                        "type": "object",
                        "description": "Alert configuration",
                        "properties": {
                            "check_interval": {
                                "type": "integer",
                                "description": "Alert check interval in seconds",
                                "minimum": 60,
                                "default": 300
                            },
                            "max_alerts": {
                                "type": "integer",
                                "description": "Maximum number of active alerts",
                                "minimum": 1,
                                "default": 50
                            }
                        }
                    },
                    "metrics": {
                        "type": "object",
                        "description": "Metric configuration",
                        "properties": {
                            "update_interval": {
                                "type": "integer",
                                "description": "Metric update interval in seconds",
                                "minimum": 60,
                                "default": 3600
                            },
                            "max_history": {
                                "type": "integer",
                                "description": "Maximum number of historical values per metric",
                                "minimum": 1,
                                "default": 100
                            }
                        }
                    }
                }
            }
        )
    
    async def initialize(self) -> bool:
        """Initialize plugin."""
        try:
            # Initialize widgets
            self.widgets = {
                "sentiment_trends": {
                    "id": "sentiment_trends",
                    "type": "line_chart",
                    "title": "Sentiment Trends",
                    "description": "Track sentiment changes over time",
                    "config": {
                        "metrics": ["positive_sentiment", "negative_sentiment"],
                        "time_range": "7d",
                        "interval": "1d"
                    }
                },
                "mention_volume": {
                    "id": "mention_volume",
                    "type": "bar_chart",
                    "title": "Mention Volume",
                    "description": "Track mention volume by source",
                    "config": {
                        "metrics": ["total_mentions"],
                        "group_by": "source",
                        "time_range": "7d"
                    }
                },
                "top_mentions": {
                    "id": "top_mentions",
                    "type": "table",
                    "title": "Top Mentions",
                    "description": "Most significant mentions",
                    "config": {
                        "metrics": ["engagement", "reach"],
                        "limit": 10,
                        "sort_by": "engagement"
                    }
                },
                "sentiment_distribution": {
                    "id": "sentiment_distribution",
                    "type": "pie_chart",
                    "title": "Sentiment Distribution",
                    "description": "Distribution of sentiment across mentions",
                    "config": {
                        "metrics": ["sentiment"],
                        "group_by": "sentiment"
                    }
                },
                "media_reach": {
                    "id": "media_reach",
                    "type": "gauge",
                    "title": "Media Reach",
                    "description": "Total reach across media channels",
                    "config": {
                        "metrics": ["total_reach"],
                        "thresholds": {
                            "low": 1000,
                            "medium": 5000,
                            "high": 10000
                        }
                    }
                }
            }
            
            # Initialize alerts
            self.alerts = {
                "negative_coverage": {
                    "id": "negative_coverage",
                    "name": "Negative Media Coverage",
                    "description": "Alert on significant negative media coverage",
                    "conditions": [
                        {
                            "metric": "negative_sentiment",
                            "operator": ">",
                            "threshold": 0.7,
                            "duration": "1h"
                        }
                    ],
                    "severity": "high",
                    "actions": [
                        {
                            "type": "notification",
                            "channels": ["email", "slack"]
                        }
                    ]
                },
                "high_volume": {
                    "id": "high_volume",
                    "name": "High Volume Mentions",
                    "description": "Alert on unusually high mention volume",
                    "conditions": [
                        {
                            "metric": "mention_volume",
                            "operator": ">",
                            "threshold": 100,
                            "duration": "1h"
                        }
                    ],
                    "severity": "medium",
                    "actions": [
                        {
                            "type": "notification",
                            "channels": ["slack"]
                        }
                    ]
                },
                "viral_content": {
                    "id": "viral_content",
                    "name": "Viral Content",
                    "description": "Alert on content going viral",
                    "conditions": [
                        {
                            "metric": "engagement_rate",
                            "operator": ">",
                            "threshold": 0.1,
                            "duration": "1h"
                        }
                    ],
                    "severity": "high",
                    "actions": [
                        {
                            "type": "notification",
                            "channels": ["email", "slack"]
                        }
                    ]
                }
            }
            
            # Initialize metrics
            self.metrics = {
                "media_sentiment": {
                    "id": "media_sentiment",
                    "name": "Media Sentiment",
                    "description": "Overall sentiment in media coverage",
                    "weight": 0.3,
                    "thresholds": {
                        "excellent": 0.8,
                        "good": 0.6,
                        "average": 0.4,
                        "poor": 0.2
                    }
                },
                "media_reach": {
                    "id": "media_reach",
                    "name": "Media Reach",
                    "description": "Total reach of media coverage",
                    "weight": 0.3,
                    "thresholds": {
                        "excellent": 100000,
                        "good": 50000,
                        "average": 10000,
                        "poor": 1000
                    }
                },
                "engagement": {
                    "id": "engagement",
                    "name": "Engagement",
                    "description": "Audience engagement with content",
                    "weight": 0.2,
                    "thresholds": {
                        "excellent": 0.1,
                        "good": 0.05,
                        "average": 0.02,
                        "poor": 0.01
                    }
                },
                "virality": {
                    "id": "virality",
                    "name": "Virality",
                    "description": "Content sharing and amplification",
                    "weight": 0.2,
                    "thresholds": {
                        "excellent": 0.2,
                        "good": 0.1,
                        "average": 0.05,
                        "poor": 0.01
                    }
                }
            }
            
            return True
            
        except Exception as e:
            raise ReputationError(
                message=f"Error initializing PR agency use case: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    async def shutdown(self) -> bool:
        """Shutdown plugin."""
        try:
            self.widgets.clear()
            self.alerts.clear()
            self.metrics.clear()
            return True
        except Exception as e:
            raise ReputationError(
                message=f"Error shutting down PR agency use case: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    async def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard configuration."""
        try:
            return {
                "widgets": list(self.widgets.values()),
                "layout": {
                    "type": "grid",
                    "columns": 3,
                    "rows": 4
                },
                "refresh_interval": self.config.get("dashboard", {}).get("refresh_interval", 300)
            }
        except Exception as e:
            raise ReputationError(
                message=f"Error getting dashboard config: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    async def get_alert_config(self) -> Dict[str, Any]:
        """Get alert configuration."""
        try:
            return {
                "alerts": list(self.alerts.values()),
                "check_interval": self.config.get("alerts", {}).get("check_interval", 300),
                "max_alerts": self.config.get("alerts", {}).get("max_alerts", 50)
            }
        except Exception as e:
            raise ReputationError(
                message=f"Error getting alert config: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    async def get_scoring_config(self) -> Dict[str, Any]:
        """Get scoring configuration."""
        try:
            return {
                "metrics": list(self.metrics.values()),
                "update_interval": self.config.get("metrics", {}).get("update_interval", 3600),
                "max_history": self.config.get("metrics", {}).get("max_history", 100)
            }
        except Exception as e:
            raise ReputationError(
                message=f"Error getting scoring config: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            ) 
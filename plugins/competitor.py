"""
Competitor monitoring plugin.
Tracks and benchmarks against competing brands.
"""

from typing import Dict, List, Any, Optional, Union, TypedDict
import asyncio
from datetime import datetime, timedelta, timezone
from app.core.plugins.base import CompetitorPlugin, PluginType, PluginMetadata
from app.core.error_handling import ReputationError, ErrorSeverity, ErrorCategory

class CompetitorData(TypedDict):
    """Type definition for competitor data."""
    name: str
    domains: List[str]
    social_handles: Dict[str, str]
    metrics: List[str]
    added_at: datetime
    last_updated: Optional[datetime]

class MetricValue(TypedDict):
    """Type definition for metric value."""
    value: float
    timestamp: datetime

class ReputationCompetitor(CompetitorPlugin):
    """Plugin for competitor monitoring and benchmarking."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize competitor monitoring plugin."""
        super().__init__(config)
        self.competitors: Dict[str, CompetitorData] = {}
        self.metrics: Dict[str, Dict[str, List[MetricValue]]] = {}
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="reputation_competitor",
            version="1.0.0",
            description="Competitor monitoring and benchmarking plugin",
            author="Reputation Sync Team",
            type=PluginType.COMPETITOR,
            config_schema={
                "type": "object",
                "properties": {
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Metrics to track for competitors"
                    },
                    "update_frequency": {
                        "type": "integer",
                        "description": "Update frequency in minutes",
                        "minimum": 1
                    },
                    "benchmark_thresholds": {
                        "type": "object",
                        "description": "Thresholds for benchmarking",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "high": {"type": "number"},
                                "low": {"type": "number"}
                            }
                        }
                    },
                    "max_history": {
                        "type": "integer",
                        "description": "Maximum number of historical values to keep per metric",
                        "default": 100
                    }
                }
            }
        )
    
    async def initialize(self) -> bool:
        """Initialize plugin."""
        return True
    
    async def shutdown(self) -> bool:
        """Shutdown plugin."""
        return True
    
    async def add_competitor(
        self,
        name: str,
        domains: List[str],
        social_handles: Dict[str, str],
        metrics: Optional[List[str]] = None
    ) -> str:
        """Add a competitor to monitor."""
        try:
            # Generate competitor ID
            competitor_id = f"comp_{len(self.competitors)}"
            
            # Add competitor
            self.competitors[competitor_id] = {
                "name": name,
                "domains": domains,
                "social_handles": social_handles,
                "metrics": metrics or self.config.get("metrics", []),
                "added_at": datetime.now(timezone.utc),
                "last_updated": None
            }
            
            # Initialize metrics
            self.metrics[competitor_id] = {
                metric: [] for metric in self.competitors[competitor_id]["metrics"]
            }
            
            return competitor_id
            
        except Exception as e:
            raise ReputationError(
                message=f"Error adding competitor: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    async def update_competitor_metrics(
        self,
        competitor_id: str
    ) -> bool:
        """Update metrics for a competitor."""
        try:
            if competitor_id not in self.competitors:
                raise ReputationError(
                    message="Competitor not found",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            competitor = self.competitors[competitor_id]
            max_history = self.config.get("max_history", 100)
            
            # Update metrics
            for metric in competitor["metrics"]:
                value = await self._fetch_metric(competitor, metric)
                timestamp = datetime.now(timezone.utc)
                
                # Add new value
                self.metrics[competitor_id][metric].append({
                    "value": value,
                    "timestamp": timestamp
                })
                
                # Trim history if needed
                if len(self.metrics[competitor_id][metric]) > max_history:
                    self.metrics[competitor_id][metric] = self.metrics[competitor_id][metric][-max_history:]
            
            # Update last updated timestamp
            competitor["last_updated"] = datetime.now(timezone.utc)
            
            return True
            
        except Exception as e:
            raise ReputationError(
                message=f"Error updating competitor metrics: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    async def get_competitor_metrics(
        self,
        competitor_id: str
    ) -> Dict[str, Any]:
        """Get metrics for a competitor."""
        try:
            if competitor_id not in self.competitors:
                raise ReputationError(
                    message="Competitor not found",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            competitor = self.competitors[competitor_id]
            metrics = self.metrics.get(competitor_id, {})
            
            return {
                "competitor_id": competitor_id,
                "name": competitor["name"],
                "metrics": metrics,
                "last_updated": competitor["last_updated"]
            }
            
        except Exception as e:
            raise ReputationError(
                message=f"Error getting competitor metrics: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    async def compare_competitors(
        self,
        competitor_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare metrics across competitors."""
        try:
            # Validate competitors
            for competitor_id in competitor_ids:
                if competitor_id not in self.competitors:
                    raise ReputationError(
                        message=f"Competitor not found: {competitor_id}",
                        severity=ErrorSeverity.HIGH,
                        category=ErrorCategory.BUSINESS
                    )
            
            # Get metrics to compare
            metrics_to_compare = metrics or self.config.get("metrics", [])
            
            # Collect metrics
            comparison = {
                "competitors": {},
                "benchmarks": {},
                "rankings": {}
            }
            
            for competitor_id in competitor_ids:
                competitor = self.competitors[competitor_id]
                competitor_metrics = self.metrics.get(competitor_id, {})
                
                comparison["competitors"][competitor_id] = {
                    "name": competitor["name"],
                    "metrics": {
                        metric: competitor_metrics.get(metric)
                        for metric in metrics_to_compare
                    }
                }
            
            # Calculate benchmarks
            for metric in metrics_to_compare:
                values = [
                    comp["metrics"][metric]
                    for comp in comparison["competitors"].values()
                    if comp["metrics"][metric] is not None
                ]
                
                if values:
                    comparison["benchmarks"][metric] = {
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "median": sorted(values)[len(values) // 2]
                    }
                    
                    # Calculate rankings
                    rankings = []
                    for competitor_id, comp in comparison["competitors"].items():
                        value = comp["metrics"][metric]
                        if value is not None:
                            rankings.append((competitor_id, value))
                    
                    rankings.sort(key=lambda x: x[1], reverse=True)
                    comparison["rankings"][metric] = {
                        competitor_id: rank + 1
                        for rank, (competitor_id, _) in enumerate(rankings)
                    }
            
            return comparison
            
        except Exception as e:
            raise ReputationError(
                message=f"Error comparing competitors: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    async def get_competitor_insights(
        self,
        competitor_id: str
    ) -> Dict[str, Any]:
        """Get insights about a competitor."""
        try:
            if competitor_id not in self.competitors:
                raise ReputationError(
                    message="Competitor not found",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            competitor = self.competitors[competitor_id]
            metrics = self.metrics.get(competitor_id, {})
            
            # Calculate insights
            insights = {
                "strengths": [],
                "weaknesses": [],
                "opportunities": [],
                "threats": []
            }
            
            # Analyze metrics against thresholds
            thresholds = self.config.get("benchmark_thresholds", {})
            for metric, value in metrics.items():
                if metric in thresholds:
                    threshold = thresholds[metric]
                    
                    if value >= threshold.get("high", float("inf")):
                        insights["strengths"].append({
                            "metric": metric,
                            "value": value,
                            "threshold": threshold["high"]
                        })
                    elif value <= threshold.get("low", float("-inf")):
                        insights["weaknesses"].append({
                            "metric": metric,
                            "value": value,
                            "threshold": threshold["low"]
                        })
            
            # Identify opportunities and threats
            for metric in competitor["metrics"]:
                if metric not in metrics:
                    insights["opportunities"].append({
                        "metric": metric,
                        "reason": "Metric not being tracked"
                    })
            
            return {
                "competitor_id": competitor_id,
                "name": competitor["name"],
                "insights": insights
            }
            
        except Exception as e:
            raise ReputationError(
                message=f"Error getting competitor insights: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    async def _fetch_metric(
        self,
        competitor: Dict[str, Any],
        metric: str
    ) -> Any:
        """Fetch a metric value for a competitor."""
        # Simulate metric fetching
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Return random value for demonstration
        return 0.0 
"""
Base plugin system.
Provides abstract classes for different types of plugins.
"""

from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel, Field


class PluginType(str, Enum):
    """Types of plugins supported by the system."""

    ALERT = "alert"
    REPORT = "report"
    PLATFORM = "platform"
    USE_CASE = "use_case"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    AUTO_DISCOVERY = "auto_discovery"
    INTEGRATION = "integration"
    AB_TESTING = "ab_testing"
    SEO = "seo"
    COMPETITOR = "competitor"


class PluginMetadata(BaseModel):
    """Metadata for a plugin."""

    name: str = Field(..., description="Name of the plugin")
    version: str = Field(..., description="Version of the plugin")
    description: str = Field(..., description="Description of the plugin")
    author: str = Field(..., description="Author of the plugin")
    type: PluginType = Field(..., description="Type of the plugin")
    config_schema: Dict[str, Any] = Field(
        ..., description="Configuration schema for the plugin"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="List of plugin dependencies"
    )


class BasePlugin(Protocol):
    """Base plugin protocol."""

    async def initialize(self) -> bool:
        """Initialize plugin."""
        ...

    async def shutdown(self) -> bool:
        """Shutdown plugin."""
        ...

    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        ...


class AlertPlugin(BasePlugin):
    """Base class for alert plugins."""

    @abstractmethod
    async def check_alert_conditions(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check if alert conditions are met."""

    @abstractmethod
    async def send_alert(self, alert: Dict[str, Any]) -> bool:
        """Send alert notification."""


class ReportPlugin(BasePlugin):
    """Base class for report plugins."""

    @abstractmethod
    async def generate_report(
        self, data: Dict[str, Any], format: str = "pdf"
    ) -> bytes:
        """Generate report in specified format."""

    @abstractmethod
    async def get_report_templates(self) -> List[Dict[str, Any]]:
        """Get available report templates."""


class PlatformPlugin(BasePlugin):
    """Base class for platform integration plugins."""

    @abstractmethod
    async def fetch_data(
        self,
        query: Dict[str, Any],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch data from platform."""

    @abstractmethod
    async def post_data(self, data: Dict[str, Any]) -> bool:
        """Post data to platform."""


class UseCasePlugin(BasePlugin):
    """Base class for use case plugins."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize use case plugin."""
        self.config = config

    @abstractmethod
    async def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard configuration."""

    @abstractmethod
    async def get_alert_config(self) -> Dict[str, Any]:
        """Get alert configuration."""

    @abstractmethod
    async def get_scoring_config(self) -> Dict[str, Any]:
        """Get scoring configuration."""


class KnowledgeGraphPlugin(BasePlugin):
    """Base class for knowledge graph plugins."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize knowledge graph plugin."""
        self.config = config

    @abstractmethod
    async def add_entity(
        self, entity_type: str, entity_data: Dict[str, Any]
    ) -> str:
        """Add entity to knowledge graph."""

    @abstractmethod
    async def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add relationship to knowledge graph."""

    @abstractmethod
    async def query_entities(
        self, query: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Query entities in knowledge graph."""

    @abstractmethod
    async def get_entity_relationships(
        self, entity_id: str, relationship_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get relationships for an entity."""


class AutoDiscoveryPlugin(BasePlugin):
    """Base class for auto-discovery plugins."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize auto-discovery plugin."""
        self.config = config

    @abstractmethod
    async def discover_handles(self, brand_name: str) -> List[Dict[str, Any]]:
        """Discover social media handles."""

    @abstractmethod
    async def discover_patterns(self, brand_name: str) -> List[Dict[str, Any]]:
        """Discover mention patterns."""


class IntegrationPlugin(BasePlugin):
    """Base class for two-way integration plugins."""

    @abstractmethod
    async def get_actions(self) -> List[Dict[str, Any]]:
        """Get available actions."""

    @abstractmethod
    async def execute_action(
        self, action: str, params: Dict[str, Any]
    ) -> bool:
        """Execute action on platform."""


class ABTestingPlugin(BasePlugin):
    """Base class for A/B testing plugins."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize A/B testing plugin."""
        self.config = config

    @abstractmethod
    async def create_experiment(
        self, name: str, variants: List[str], metrics: List[str]
    ) -> str:
        """Create A/B test experiment."""

    @abstractmethod
    async def assign_variant(self, experiment_id: str, user_id: str) -> str:
        """Assign user to variant."""

    @abstractmethod
    async def record_metric(
        self, experiment_id: str, user_id: str, metric: str, value: float
    ) -> bool:
        """Record metric value."""

    @abstractmethod
    async def get_experiment_results(
        self, experiment_id: str
    ) -> Dict[str, Any]:
        """Get experiment results."""


class SEOPlugin(BasePlugin):
    """Base class for SEO plugins."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize SEO plugin."""
        self.config = config

    @abstractmethod
    async def analyze_url(self, url: str) -> Dict[str, Any]:
        """Analyze URL for SEO metrics."""

    @abstractmethod
    async def track_keyword_rankings(
        self, keywords: List[str], search_engines: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Track keyword rankings."""

    @abstractmethod
    async def analyze_competitor(self, url: str) -> Dict[str, Any]:
        """Analyze competitor's SEO performance."""


class CompetitorPlugin(BasePlugin):
    """Base class for competitor monitoring plugins."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize competitor plugin."""
        self.config = config

    @abstractmethod
    async def add_competitor(
        self, name: str, url: str, metrics: List[str]
    ) -> str:
        """Add competitor to monitor."""

    @abstractmethod
    async def update_competitor_metrics(
        self, competitor_id: str, metrics: Dict[str, float]
    ) -> bool:
        """Update competitor metrics."""

    @abstractmethod
    async def get_competitor_metrics(
        self, competitor_id: str, metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get competitor metrics."""

    @abstractmethod
    async def compare_competitors(
        self, competitor_ids: List[str], metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare competitors."""

"""
AI configuration management.
Provides centralized configuration for AI components.
"""

from typing import Any, Dict

from app.core.config import get_settings

settings = get_settings()


class AIConfig:
    """AI configuration management."""

    def __init__(self):
        """Initialize AI configuration."""
        self.settings = settings

    def get_continuous_config(self) -> Dict[str, Any]:
        """Get continuous processing configuration."""
        return {
            "enabled": self.settings.CONTINUOUS_PROCESSING_ENABLED,
            "interval": self.settings.PROCESSING_INTERVAL,
            "health_check_interval": self.settings.HEALTH_CHECK_INTERVAL,
        }

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "path": self.settings.MODEL_PATH,
            "cache_size": self.settings.MODEL_CACHE_SIZE,
            "max_pipelines": self.settings.MAX_CONCURRENT_PIPELINES,
        }


# Global instance
ai_config = AIConfig()

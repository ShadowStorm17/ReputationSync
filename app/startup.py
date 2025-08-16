"""
Advanced application startup system.
Provides comprehensive initialization, validation, and health checks.
"""

import asyncio
import logging
from datetime import datetime, timezone

from prometheus_client import Counter, Gauge, Histogram

from app.core.caching import cache
from app.core.config import get_settings
from app.core.error_handling import error_handler, SystemError
from app.core.metrics import metrics_manager
from app.core.orchestrator import orchestrator
from app.core.rate_limiting import rate_limiter
from app.core.security import security_manager
from app.services.enhanced_predictive import EnhancedPredictive
from app.services.enhanced_response import EnhancedResponse
from app.services.sentiment_service import SentimentService

# Startup metrics
STARTUP_TIME = Histogram(
    'startup_time_seconds',
    'Component startup time',
    ['component'])
STARTUP_ERRORS = Counter(
    'startup_errors_total',
    'Startup errors',
    ['component'])
COMPONENT_READY = Gauge(
    'component_ready',
    'Component ready status',
    ['component'])

logger = logging.getLogger(__name__)
settings = get_settings()


class StartupManager:
    """Advanced startup management."""

    def __init__(self):
        """Initialize startup manager."""
        self.startup_order = [
            "error_handler",
            "security",
            "cache",
            "rate_limiter",
            "metrics",
            "orchestrator",
            "sentiment",
            "predictive",
            "response"
        ]
        self.components = {
            "error_handler": error_handler,
            "security": security_manager,
            "cache": cache,
            "rate_limiter": rate_limiter,
            "metrics": metrics_manager,
            "orchestrator": orchestrator,
            "sentiment": SentimentService(),
            "predictive": EnhancedPredictive(),
            "response": EnhancedResponse()
        }
        self.ready_components = set()

    async def start(self):
        """Start all components in order."""
        logger.info("Starting application components...")

        try:
            # Initialize error handling first
            await self._start_component("error_handler")

            # Start remaining components in order
            for component_name in self.startup_order[1:]:
                await self._start_component(component_name)

            # Verify all components
            await self._verify_components()

            logger.info("All components started successfully")

        except SystemError as e:
            logger.error("Startup failed: %s", str(e))
            raise

    async def _start_component(self, name: str):
        """Start a single component."""
        try:
            logger.info("Starting component: %s", name)
            start_time = datetime.now(timezone.utc)

            component = self.components[name]

            # Initialize component
            if hasattr(component, "initialize"):
                await component.initialize()

            # Verify component health
            if hasattr(component, "check_health"):
                is_healthy = await component.check_health()
                if not is_healthy:
                    raise SystemError(f"Component {name} failed health check")

            # Update metrics
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            STARTUP_TIME.labels(component=name).observe(duration)
            COMPONENT_READY.labels(component=name).set(1)

            self.ready_components.add(name)
            logger.info("Component %s started successfully", name)

        except SystemError as e:
            logger.error("Error starting component %s: %s", name, str(e))
            STARTUP_ERRORS.labels(component=name).inc()
            COMPONENT_READY.labels(component=name).set(0)
            raise

    async def _verify_components(self):
        """Verify all components are ready."""
        try:
            # Check all components are ready
            missing_components = set(
                self.startup_order) - self.ready_components
            if missing_components:
                raise SystemError(f"Components not ready: {missing_components}")

            # Verify component interactions
            await self._verify_component_interactions()

            # Verify system health
            await self._verify_system_health()

        except SystemError as e:
            logger.error("Component verification failed: %s", str(e))
            raise

    async def _verify_component_interactions(self):
        """Verify component interactions."""
        try:
            # Verify security
            token = await security_manager.create_token({
                "test": True,
                "scope": "system"
            })
            await security_manager.validate_token(token)

            # Verify cache
            await cache.set("startup_test", "test")
            await cache.get("startup_test")
            await cache.delete("startup_test")

            # Verify rate limiter
            await rate_limiter.check_rate_limit("startup_test")

            # Verify metrics
            metrics_manager.track_metric("startup_test", 1.0)

        except SystemError as e:
            logger.error("Component interaction verification failed: %s", str(e))
            raise

    async def _verify_system_health(self):
        """Verify overall system health."""
        try:
            # Get component health status
            health_status = await orchestrator.get_health_status()

            # Check all components are healthy
            unhealthy_components = [
                name for name, status in health_status.items()
                if not status["healthy"]
            ]

            if unhealthy_components:
                raise SystemError(
                    f"Unhealthy components: {unhealthy_components}")

            logger.info("System health verification completed successfully")

        except SystemError as e:
            logger.error("System health verification failed: %s", str(e))
            raise


def run():
    """Run the application."""
    try:
        # Create and set event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Run startup process
        startup_manager = StartupManager()
        loop.run_until_complete(startup_manager.start())

        logger.info("Application started successfully")

        # Keep the loop running
        loop.run_forever()

    except KeyboardInterrupt:
        logger.info("Shutting down application...")
        loop.stop()
        loop.close()
    except SystemError as e:
        logger.error("Application startup failed: %s", str(e))
        raise
    finally:
        if loop.is_running():
            loop.stop()
        if not loop.is_closed():
            loop.close()


if __name__ == "__main__":
    run()

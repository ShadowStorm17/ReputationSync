"""
Advanced orchestration system for managing all API components.
Provides intelligent routing, load balancing, and automatic failover.
"""

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
from datetime import timezone

from prometheus_client import Counter, Gauge, Histogram

from app.core.config import get_settings
from app.core.error_handling import ErrorCategory, ErrorSeverity, handle_errors
from app.core.pipeline_manager import pipeline_manager
from app.services.enhanced_predictive import EnhancedPredictive
from app.services.enhanced_response import EnhancedResponse
from app.services.sentiment_service import SentimentService

# Enhanced metrics
COMPONENT_HEALTH = Gauge(
    'component_health',
    'Component health status',
    ['component'])
COMPONENT_LATENCY = Histogram(
    'component_latency_seconds',
    'Component operation latency',
    ['component'])
FAILOVER_COUNT = Counter('failover_total', 'Failover count', ['component'])
RECOVERY_TIME = Histogram(
    'recovery_time_seconds',
    'Component recovery time',
    ['component'])

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ComponentState:
    """Component state tracking."""
    is_healthy: bool = True
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_count: int = 0
    recovery_attempts: int = 0
    latency: float = 0.0
    load: float = 0.0


class Orchestrator:
    """Advanced orchestration system."""

    def __init__(self):
        """Initialize orchestrator."""
        self._lock = threading.Lock()
        self._components: Dict[str, Any] = {}
        self._states: Dict[str, ComponentState] = {}
        self._initialize_components()
        self._start_background_tasks()

    def _initialize_components(self):
        """Initialize all core components."""
        try:
            # Initialize core services
            self._components["monitoring"] = pipeline_manager
            self._components["sentiment"] = SentimentService()
            self._components["predictive"] = EnhancedPredictive()
            self._components["response"] = EnhancedResponse()

            # Initialize states
            for name in self._components:
                self._states[name] = ComponentState()

            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Component initialization error: {str(e)}")
            raise

    async def _health_check(self):
        """Continuous health monitoring."""
        while True:
            try:
                for name, component in self._components.items():
                    state = self._states[name]

                    try:
                        # Check component health
                        start_time = datetime.now(timezone.utc)
                        is_healthy = await self._check_component_health(component)
                        latency = (
                            datetime.now(timezone.utc) -
                            start_time).total_seconds()

                        # Update state
                        state.is_healthy = is_healthy
                        state.latency = latency
                        state.last_check = datetime.now(timezone.utc)

                        if is_healthy:
                            state.error_count = 0
                            state.recovery_attempts = 0
                        else:
                            state.error_count += 1
                            await self._handle_unhealthy_component(name, component, state)

                        # Update metrics
                        COMPONENT_HEALTH.labels(
                            component=name).set(
                            1 if is_healthy else 0)
                        COMPONENT_LATENCY.labels(
                            component=name).observe(latency)

                    except Exception as e:
                        logger.error(
                            f"Health check error for {name}: {str(e)}"
                        )
                        state.error_count += 1

                await asyncio.sleep(settings.HEALTH_CHECK_INTERVAL)

            except Exception as e:
                logger.error(f"Health check loop error: {str(e)}")
                await asyncio.sleep(60)

    async def _check_component_health(self, component: Any) -> bool:
        """Check health of a component."""
        try:
            # Call component's health check method if available
            if hasattr(component, "check_health"):
                return await component.check_health()

            # Default health check
            return True

        except Exception as e:
            logger.error(f"Component health check error: {str(e)}")
            return False

    async def _handle_unhealthy_component(
        self,
        name: str,
        component: Any,
        state: ComponentState
    ):
        """Handle unhealthy component."""
        try:
            # Attempt recovery
            if state.error_count >= settings.MAX_ERRORS_BEFORE_RECOVERY:
                state.recovery_attempts += 1

                start_time = datetime.now(timezone.utc)
                success = await self._recover_component(name, component)
                recovery_time = (
                    datetime.now(timezone.utc) -
                    start_time).total_seconds()

                if success:
                    logger.info(f"Successfully recovered component {name}")
                    state.error_count = 0
                    RECOVERY_TIME.labels(component=name).observe(recovery_time)
                else:
                    logger.error(f"Failed to recover component {name}")
                    if state.recovery_attempts >= settings.MAX_RECOVERY_ATTEMPTS:
                        await self._failover_component(name, component)

        except Exception as e:
            logger.error(
                f"Error handling unhealthy component {name}: {str(e)}"
            )

    async def _recover_component(self, name: str, component: Any) -> bool:
        """Attempt to recover a component."""
        try:
            # Call component's recovery method if available
            if hasattr(component, "recover"):
                return await component.recover()

            # Default recovery - reinitialize component
            self._components[name] = type(component)()
            return True

        except Exception as e:
            logger.error(f"Component recovery error: {str(e)}")
            return False

    async def _failover_component(self, name: str, component: Any):
        """Failover to backup component."""
        try:
            # Increment failover counter
            FAILOVER_COUNT.labels(component=name).inc()

            # Create new instance
            new_component = type(component)()

            # Swap components
            with self._lock:
                self._components[name] = new_component

            logger.info(f"Successfully failed over component {name}")

        except Exception as e:
            logger.error(f"Failover error for component {name}: {str(e)}")

    def _start_background_tasks(self):
        """Start background tasks."""
        asyncio.create_task(self._health_check())

    @handle_errors(ErrorSeverity.HIGH, ErrorCategory.SYSTEM)
    async def get_component(self, name: str) -> Optional[Any]:
        """Get a component by name."""
        return self._components.get(name)

    @handle_errors(ErrorSeverity.HIGH, ErrorCategory.SYSTEM)
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all components."""
        status = {}
        for name, state in self._states.items():
            status[name] = {
                "healthy": state.is_healthy,
                "last_check": state.last_check.isoformat(),
                "error_count": state.error_count,
                "latency": state.latency
            }
        return status


# Global orchestrator instance
orchestrator = Orchestrator()

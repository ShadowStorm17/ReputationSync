"""
Service discovery service for service registration and discovery.
Provides service registry and health monitoring.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Optional

import aiohttp

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ServiceStatus(str, Enum):
    """Service status states."""
    UNKNOWN = 'unknown'
    STARTING = 'starting'
    RUNNING = 'running'
    STOPPING = 'stopping'
    STOPPED = 'stopped'
    FAILED = 'failed'


class ServiceInstance:
    """Service instance representation."""

    def __init__(
        self,
        instance_id: str,
        service_name: str,
        host: str,
        port: int,
        metadata: Optional[Dict[str, Any]] = None,
        health_check_url: Optional[str] = None,
        health_check_interval: int = 30
    ):
        """Initialize service instance."""
        self.instance_id = instance_id
        self.service_name = service_name
        self.host = host
        self.port = port
        self.metadata = metadata or {}
        self.health_check_url = health_check_url
        self.health_check_interval = health_check_interval
        self.status = ServiceStatus.UNKNOWN
        self.last_check = None
        self.registered_at = datetime.now(timezone.utc)
        self.last_heartbeat = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to dictionary."""
        return {
            'instance_id': self.instance_id,
            'service_name': self.service_name,
            'host': self.host,
            'port': self.port,
            'metadata': self.metadata,
            'health_check_url': self.health_check_url,
            'health_check_interval': self.health_check_interval,
            'status': self.status,
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'registered_at': self.registered_at.isoformat(),
            'last_heartbeat': self.last_heartbeat.isoformat()}


class ServiceRegistry:
    """Service registry implementation."""

    def __init__(
        self,
        heartbeat_timeout: int = 60,
        cleanup_interval: int = 300
    ):
        """Initialize registry."""
        self.heartbeat_timeout = heartbeat_timeout
        self.cleanup_interval = cleanup_interval
        self.services: Dict[str, Dict[str, ServiceInstance]] = {}
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        self.running = False

    async def register_instance(
        self,
        instance: ServiceInstance
    ) -> bool:
        """Register service instance."""
        try:
            if instance.service_name not in self.services:
                self.services[instance.service_name] = {}

            self.services[instance.service_name][instance.instance_id] = instance

            # Perform initial health check
            if instance.health_check_url:
                await self.check_instance_health(instance)
            else:
                instance.status = ServiceStatus.RUNNING

            return True

        except Exception as e:
            logger.error("Register instance error: %s", e)
            return False

    async def deregister_instance(
        self,
        service_name: str,
        instance_id: str
    ) -> bool:
        """Deregister service instance."""
        try:
            if service_name not in self.services:
                return False

            if instance_id not in self.services[service_name]:
                return False

            instance = self.services[service_name][instance_id]
            instance.status = ServiceStatus.STOPPING

            del self.services[service_name][instance_id]

            if not self.services[service_name]:
                del self.services[service_name]

            return True

        except Exception as e:
            logger.error("Deregister instance error: %s", e)
            return False

    async def heartbeat(
        self,
        service_name: str,
        instance_id: str
    ) -> bool:
        """Update instance heartbeat."""
        try:
            if service_name not in self.services:
                return False

            if instance_id not in self.services[service_name]:
                return False

            instance = self.services[service_name][instance_id]
            instance.last_heartbeat = datetime.now(timezone.utc)

            return True

        except Exception as e:
            logger.error("Heartbeat error: %s", e)
            return False

    async def check_instance_health(
        self,
        instance: ServiceInstance
    ):
        """Check instance health."""
        try:
            if not instance.health_check_url:
                return

            async with self.session.get(
                instance.health_check_url,
                timeout=5
            ) as response:
                instance.last_check = datetime.now(timezone.utc)

                if response.status == 200:
                    instance.status = ServiceStatus.RUNNING
                else:
                    instance.status = ServiceStatus.FAILED

        except Exception as e:
            logger.error(
                "Health check error for %s: %s",
                instance.instance_id, e)
            instance.status = ServiceStatus.FAILED

    async def cleanup_expired_instances(self):
        """Remove expired instances."""
        try:
            now = datetime.now(timezone.utc)
            expired_timeout = timedelta(seconds=self.heartbeat_timeout)

            for service_name in list(self.services.keys()):
                for instance_id in list(self.services[service_name].keys()):
                    instance = self.services[service_name][instance_id]

                    if now - instance.last_heartbeat > expired_timeout:
                        await self.deregister_instance(
                            service_name,
                            instance_id
                        )

        except Exception as e:
            logger.error("Cleanup error: %s", e)

    async def start_health_checks(self):
        """Start health check and cleanup tasks."""
        self.running = True

        while self.running:
            try:
                # Perform health checks
                for service_instances in self.services.values():
                    for instance in service_instances.values():
                        if instance.health_check_url:
                            await self.check_instance_health(instance)

                # Cleanup expired instances
                await self.cleanup_expired_instances()

                await asyncio.sleep(self.cleanup_interval)

            except Exception as e:
                logger.error("Health check loop error: %s", e)
                await asyncio.sleep(self.cleanup_interval)

    def stop_health_checks(self):
        """Stop health check task."""
        self.running = False

    async def close(self):
        """Close registry."""
        self.stop_health_checks()
        await self.session.close()


class DiscoveryService:
    """Service discovery management service."""

    def __init__(self):
        """Initialize discovery service."""
        self.registry = ServiceRegistry()

    async def register_service(
        self,
        instance_id: str,
        service_name: str,
        host: str,
        port: int,
        metadata: Optional[Dict[str, Any]] = None,
        health_check_url: Optional[str] = None,
        health_check_interval: int = 30
    ) -> Dict[str, Any]:
        """Register service instance."""
        try:
            instance = ServiceInstance(
                instance_id,
                service_name,
                host,
                port,
                metadata,
                health_check_url,
                health_check_interval
            )

            success = await self.registry.register_instance(instance)

            return {
                'status': 'success' if success else 'error',
                'message': (
                    'Service registered successfully'
                    if success else
                    'Failed to register service'
                ),
                'instance': instance.to_dict() if success else None
            }

        except Exception as e:
            logger.error("Register service error: %s", e)
            return {
                'status': 'error',
                'message': str(e)
            }

    async def deregister_service(
        self,
        service_name: str,
        instance_id: str
    ) -> Dict[str, Any]:
        """Deregister service instance."""
        try:
            success = await self.registry.deregister_instance(
                service_name,
                instance_id
            )

            return {
                'status': 'success' if success else 'error',
                'message': (
                    'Service deregistered successfully'
                    if success else
                    'Failed to deregister service'
                )
            }

        except Exception as e:
            logger.error("Deregister service error: %s", e)
            return {
                'status': 'error',
                'message': str(e)
            }

    async def get_service_instances(
        self,
        service_name: str
    ) -> Dict[str, Any]:
        """Get service instances."""
        try:
            if service_name not in self.registry.services:
                return {
                    'status': 'error',
                    'message': f"Service not found: {service_name}"
                }

            instances = [
                instance.to_dict()
                for instance in self.registry.services[service_name].values()
                if instance.status == ServiceStatus.RUNNING
            ]

            return {
                'status': 'success',
                'instances': instances
            }

        except Exception as e:
            logger.error("Get instances error: %s", e)
            return {
                'status': 'error',
                'message': str(e)
            }

    async def heartbeat(
        self,
        service_name: str,
        instance_id: str
    ) -> Dict[str, Any]:
        """Send service heartbeat."""
        try:
            success = await self.registry.heartbeat(
                service_name,
                instance_id
            )

            return {
                'status': 'success' if success else 'error',
                'message': (
                    'Heartbeat received'
                    if success else
                    'Failed to update heartbeat'
                )
            }

        except Exception as e:
            logger.error("Heartbeat error: %s", e)
            return {
                'status': 'error',
                'message': str(e)
            }

    async def start(self):
        """Start discovery service."""
        await self.registry.start_health_checks()

    async def stop(self):
        """Stop discovery service."""
        await self.registry.close()

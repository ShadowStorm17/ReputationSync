"""
Load balancer service for request distribution.
Provides load balancing and server health monitoring.
"""

import asyncio
import logging
import random
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

import aiohttp

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ServerStatus(str, Enum):
    """Server status states."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    DISABLED = "disabled"


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies."""

    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    WEIGHTED = "weighted"
    IP_HASH = "ip_hash"


class Server:
    """Backend server representation."""

    def __init__(
        self, name: str, url: str, weight: int = 1, max_connections: int = 100
    ):
        """Initialize server."""
        self.name = name
        self.url = url
        self.weight = weight
        self.max_connections = max_connections
        self.current_connections = 0
        self.total_requests = 0
        self.failed_checks = 0
        self.last_check = None
        self.status = ServerStatus.HEALTHY
        self.metrics: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert server to dictionary."""
        return {
            "name": self.name,
            "url": self.url,
            "weight": self.weight,
            "max_connections": self.max_connections,
            "current_connections": self.current_connections,
            "total_requests": self.total_requests,
            "failed_checks": self.failed_checks,
            "last_check": self.last_check.isoformat()
            if self.last_check
            else None,
            "status": self.status,
            "metrics": self.metrics,
        }


class LoadBalancer:
    """Load balancer implementation."""

    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
        check_interval: int = 30,
        unhealthy_threshold: int = 3,
    ):
        """Initialize load balancer."""
        self.strategy = strategy
        self.check_interval = check_interval
        self.unhealthy_threshold = unhealthy_threshold
        self.servers: Dict[str, Server] = {}
        self.current_index = 0
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        self.running = False

    async def add_server(
        self, name: str, url: str, weight: int = 1, max_connections: int = 100
    ) -> bool:
        """Add server to pool."""
        try:
            if name in self.servers:
                return False

            server = Server(name, url, weight, max_connections)
            self.servers[name] = server

            # Perform initial health check
            await self.check_server_health(server)

            return True

        except Exception as e:
            logger.error("Add server error: %s", e)
            return False

    async def remove_server(self, name: str, drain: bool = True) -> bool:
        """Remove server from pool."""
        try:
            if name not in self.servers:
                return False

            server = self.servers[name]

            if drain:
                # Set status to draining
                server.status = ServerStatus.DRAINING

                # Wait for connections to complete
                while server.current_connections > 0:
                    await asyncio.sleep(1)

            del self.servers[name]
            return True

        except Exception as e:
            logger.error("Remove server error: %s", e)
            return False

    async def get_next_server(
        self, client_ip: Optional[str] = None
    ) -> Optional[Server]:
        """Get next server based on strategy."""
        try:
            available_servers = [
                s
                for s in self.servers.values()
                if s.status == ServerStatus.HEALTHY
                and s.current_connections < s.max_connections
            ]

            if not available_servers:
                return None

            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                server = available_servers[
                    self.current_index % len(available_servers)
                ]
                self.current_index += 1
                return server

            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return min(
                    available_servers, key=lambda s: s.current_connections
                )

            elif self.strategy == LoadBalancingStrategy.RANDOM:
                return random.choice(available_servers)

            elif self.strategy == LoadBalancingStrategy.WEIGHTED:
                total_weight = sum(s.weight for s in available_servers)
                r = random.uniform(0, total_weight)
                for server in available_servers:
                    r -= server.weight
                    if r <= 0:
                        return server
                return available_servers[0]

            elif self.strategy == LoadBalancingStrategy.IP_HASH:
                if not client_ip:
                    return available_servers[0]
                hash_value = sum(ord(c) for c in client_ip)
                return available_servers[hash_value % len(available_servers)]

            return available_servers[0]

        except Exception as e:
            logger.error("Get next server error: %s", e)
            return None

    async def check_server_health(self, server: Server):
        """Check server health."""
        try:
            async with self.session.get(
                f"{server.url}/health", timeout=5
            ) as response:
                server.last_check = datetime.now(timezone.utc)

                if response.status == 200:
                    metrics = await response.json()
                    server.metrics = metrics
                    server.failed_checks = 0
                    server.status = ServerStatus.HEALTHY
                else:
                    server.failed_checks += 1

                    if server.failed_checks >= self.unhealthy_threshold:
                        server.status = ServerStatus.UNHEALTHY

        except Exception as e:
            logger.error("Health check error for %s: %s", server.name, e)
            server.failed_checks += 1

            if server.failed_checks >= self.unhealthy_threshold:
                server.status = ServerStatus.UNHEALTHY

    async def start_health_checks(self):
        """Start periodic health checks."""
        self.running = True

        while self.running:
            try:
                tasks = [
                    self.check_server_health(server)
                    for server in self.servers.values()
                ]
                await asyncio.gather(*tasks)
                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error("Health check loop error: %s", e)
                await asyncio.sleep(self.check_interval)

    def stop_health_checks(self):
        """Stop periodic health checks."""
        self.running = False

    async def close(self):
        """Close load balancer."""
        self.stop_health_checks()
        await self.session.close()


class LoadBalancerService:
    """Load balancer management service."""

    def __init__(self):
        """Initialize load balancer service."""
        self.load_balancers: Dict[str, LoadBalancer] = {}

    async def create_balancer(
        self,
        name: str,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
        check_interval: int = 30,
        unhealthy_threshold: int = 3,
    ) -> LoadBalancer:
        """Create new load balancer."""
        if name not in self.load_balancers:
            balancer = LoadBalancer(
                strategy, check_interval, unhealthy_threshold
            )
            self.load_balancers[name] = balancer

            # Start health checks
            asyncio.create_task(balancer.start_health_checks())

        return self.load_balancers[name]

    def get_balancer(self, name: str) -> Optional[LoadBalancer]:
        """Get existing load balancer."""
        return self.load_balancers.get(name)

    async def add_server_to_balancer(
        self,
        balancer_name: str,
        server_name: str,
        server_url: str,
        weight: int = 1,
        max_connections: int = 100,
    ) -> Dict[str, Any]:
        """Add server to load balancer."""
        try:
            balancer = self.get_balancer(balancer_name)
            if not balancer:
                return {
                    "status": "error",
                    "message": f"Load balancer not found: {balancer_name}",
                }

            success = await balancer.add_server(
                server_name, server_url, weight, max_connections
            )

            return {
                "status": "success" if success else "error",
                "message": (
                    "Server added successfully"
                    if success
                    else "Failed to add server"
                ),
            }

        except Exception as e:
            logger.error("Add server error: %s", e)
            return {"status": "error", "message": str(e)}

    async def remove_server_from_balancer(
        self, balancer_name: str, server_name: str, drain: bool = True
    ) -> Dict[str, Any]:
        """Remove server from load balancer."""
        try:
            balancer = self.get_balancer(balancer_name)
            if not balancer:
                return {
                    "status": "error",
                    "message": f"Load balancer not found: {balancer_name}",
                }

            success = await balancer.remove_server(server_name, drain)

            return {
                "status": "success" if success else "error",
                "message": (
                    "Server removed successfully"
                    if success
                    else "Failed to remove server"
                ),
            }

        except Exception as e:
            logger.error("Remove server error: %s", e)
            return {"status": "error", "message": str(e)}

    async def get_server_metrics(self, balancer_name: str) -> Dict[str, Any]:
        """Get server metrics."""
        try:
            balancer = self.get_balancer(balancer_name)
            if not balancer:
                return {
                    "status": "error",
                    "message": f"Load balancer not found: {balancer_name}",
                }

            metrics = {
                name: server.to_dict()
                for name, server in balancer.servers.items()
            }

            return {"status": "success", "metrics": metrics}

        except Exception as e:
            logger.error("Get metrics error: %s", e)
            return {"status": "error", "message": str(e)}

    async def close_all(self):
        """Close all load balancers."""
        try:
            for balancer in self.load_balancers.values():
                await balancer.close()

            self.load_balancers.clear()

        except Exception as e:
            logger.error("Close all error: %s", e)

"""
API gateway service for request routing and transformation.
Provides API gateway functionality and request handling.
"""

import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urljoin

import aiohttp

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class Route:
    """API route representation."""

    def __init__(
        self,
        path: str,
        target: str,
        methods: List[str] = None,
        auth_required: bool = False,
        transform_request: Optional[Callable] = None,
        transform_response: Optional[Callable] = None,
        timeout: int = 30,
        cache_ttl: Optional[int] = None,
        rate_limit: Optional[Dict[str, Any]] = None,
    ):
        """Initialize route."""
        self.path = path
        self.target = target
        self.methods = methods or ["GET"]
        self.auth_required = auth_required
        self.transform_request = transform_request
        self.transform_response = transform_response
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self.rate_limit = rate_limit
        self.metrics: Dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0,
        }


class Gateway:
    """API gateway implementation."""

    def __init__(
        self,
        name: str,
        base_path: str = "/",
        auth_service: Optional[Any] = None,
        cache_service: Optional[Any] = None,
        rate_limit_service: Optional[Any] = None,
    ):
        """Initialize gateway."""
        self.name = name
        self.base_path = base_path
        self.auth_service = auth_service
        self.cache_service = cache_service
        self.rate_limit_service = rate_limit_service
        self.routes: Dict[str, Route] = {}
        self.session = aiohttp.ClientSession()
        self.middleware: List[Callable] = []

    def add_route(self, path: str, target: str, **kwargs) -> Route:
        """Add route to gateway."""
        route = Route(path, target, **kwargs)
        self.routes[path] = route
        return route

    def add_middleware(self, middleware: Callable):
        """Add middleware function."""
        self.middleware.append(middleware)

    async def handle_request(
        self,
        path: str,
        method: str,
        headers: Dict[str, str],
        data: Any = None,
        params: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Handle incoming request."""
        try:
            # Find matching route
            route = self.routes.get(path)
            if not route:
                return {"status": 404, "body": {"error": "Route not found"}}

            # Check method
            if method not in route.methods:
                return {"status": 405, "body": {"error": "Method not allowed"}}

            # Update metrics
            route.metrics["total_requests"] += 1
            start_time = datetime.utcnow()

            # Check authentication
            if route.auth_required and self.auth_service:
                auth_result = await self.auth_service.authenticate(headers)
                if not auth_result["authenticated"]:
                    route.metrics["failed_requests"] += 1
                    return {"status": 401, "body": {"error": "Unauthorized"}}

            # Check rate limit
            if route.rate_limit and self.rate_limit_service:
                rate_limit_result = (
                    await self.rate_limit_service.check_rate_limit(
                        f"{self.name}:{path}", **route.rate_limit
                    )
                )
                if not rate_limit_result["allowed"]:
                    route.metrics["failed_requests"] += 1
                    return {
                        "status": 429,
                        "body": {"error": "Too many requests"},
                    }

            # Check cache
            cache_key = None
            if route.cache_ttl and self.cache_service and method == "GET":
                cache_key = f"{self.name}:{path}:{json.dumps(params)}"
                cached_response = await self.cache_service.get(cache_key)
                if cached_response:
                    return cached_response

            # Transform request
            if route.transform_request:
                data = route.transform_request(data)

            # Apply middleware
            for middleware in self.middleware:
                headers, data = await middleware(headers, data)

            # Forward request
            target_url = urljoin(route.target, path)
            async with self.session.request(
                method,
                target_url,
                headers=headers,
                json=data,
                params=params,
                timeout=route.timeout,
            ) as response:
                response_data = await response.json()

                # Transform response
                if route.transform_response:
                    response_data = route.transform_response(response_data)

                result = {
                    "status": response.status,
                    "body": response_data,
                    "headers": dict(response.headers),
                }

                # Cache response
                if cache_key and response.status == 200:
                    await self.cache_service.set(
                        cache_key, result, route.cache_ttl
                    )

                # Update metrics
                end_time = datetime.utcnow()
                response_time = (end_time - start_time).total_seconds()

                if response.status < 400:
                    route.metrics["successful_requests"] += 1
                else:
                    route.metrics["failed_requests"] += 1

                route.metrics["average_response_time"] = (
                    route.metrics["average_response_time"]
                    * (route.metrics["total_requests"] - 1)
                    + response_time
                ) / route.metrics["total_requests"]

                return result

        except Exception as e:
            logger.error(f"Request handling error: {str(e)}")
            return {"status": 500, "body": {"error": "Internal server error"}}

    async def close(self):
        """Close gateway."""
        await self.session.close()


class GatewayService:
    """API gateway management service."""

    def __init__(self):
        """Initialize gateway service."""
        self.gateways: Dict[str, Gateway] = {}

    def create_gateway(
        self,
        name: str,
        base_path: str = "/",
        auth_service: Optional[Any] = None,
        cache_service: Optional[Any] = None,
        rate_limit_service: Optional[Any] = None,
    ) -> Gateway:
        """Create new API gateway."""
        if name not in self.gateways:
            self.gateways[name] = Gateway(
                name,
                base_path,
                auth_service,
                cache_service,
                rate_limit_service,
            )
        return self.gateways[name]

    def get_gateway(self, name: str) -> Optional[Gateway]:
        """Get existing gateway."""
        return self.gateways.get(name)

    async def add_route(
        self, gateway_name: str, path: str, target: str, **kwargs
    ) -> Dict[str, Any]:
        """Add route to gateway."""
        try:
            gateway = self.get_gateway(gateway_name)
            if not gateway:
                return {
                    "status": "error",
                    "message": f"Gateway not found: {gateway_name}",
                }

            route = gateway.add_route(path, target, **kwargs)

            return {
                "status": "success",
                "route": {
                    "path": route.path,
                    "target": route.target,
                    "methods": route.methods,
                },
            }

        except Exception as e:
            logger.error(f"Add route error: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def get_metrics(self, gateway_name: str) -> Dict[str, Any]:
        """Get gateway metrics."""
        try:
            gateway = self.get_gateway(gateway_name)
            if not gateway:
                return {
                    "status": "error",
                    "message": f"Gateway not found: {gateway_name}",
                }

            metrics = {
                path: route.metrics for path, route in gateway.routes.items()
            }

            return {"status": "success", "metrics": metrics}

        except Exception as e:
            logger.error(f"Get metrics error: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def close_all(self):
        """Close all gateways."""
        try:
            for gateway in self.gateways.values():
                await gateway.close()

            self.gateways.clear()

        except Exception as e:
            logger.error(f"Close all error: {str(e)}")

"""
API documentation service.
Provides OpenAPI/Swagger integration and documentation management.
"""

import inspect
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class APIEndpoint(BaseModel):
    """API endpoint representation."""

    path: str
    method: str
    summary: str
    description: Optional[str] = None
    parameters: List[Dict[str, Any]] = Field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    deprecated: bool = False
    security: Optional[List[Dict[str, List[str]]]] = None


class APIVersion(BaseModel):
    """API version representation."""

    version: str
    title: str
    description: str
    endpoints: List[APIEndpoint] = Field(default_factory=list)
    servers: List[Dict[str, str]] = Field(default_factory=list)
    security_schemes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentationService:
    """API documentation management service."""

    def __init__(self):
        """Initialize documentation service."""
        self.versions: Dict[str, APIVersion] = {}
        self.docs_path = Path("docs/api")
        self.docs_path.mkdir(parents=True, exist_ok=True)

    def create_version(
        self,
        version: str,
        title: str,
        description: str,
        servers: Optional[List[Dict[str, str]]] = None,
    ) -> APIVersion:
        """Create new API version."""
        if version in self.versions:
            return self.versions[version]

        self.versions[version] = APIVersion(
            version=version,
            title=title,
            description=description,
            servers=servers or [],
        )
        return self.versions[version]

    def add_endpoint(self, version: str, endpoint: APIEndpoint) -> bool:
        """Add endpoint to API version."""
        try:
            if version not in self.versions:
                return False

            self.versions[version].endpoints.append(endpoint)
            return True

        except Exception as e:
            logger.error("Add endpoint error: %s", e)
            return False

    def add_security_scheme(
        self, version: str, name: str, scheme: Dict[str, Any]
    ) -> bool:
        """Add security scheme to API version."""
        try:
            if version not in self.versions:
                return False

            self.versions[version].security_schemes[name] = scheme
            return True

        except Exception as e:
            logger.error("Add security scheme error: %s", e)
            return False

    def generate_openapi_spec(self, version: str) -> Dict[str, Any]:
        """Generate OpenAPI specification."""
        try:
            if version not in self.versions:
                return {}

            api_version = self.versions[version]

            spec = {
                "openapi": "3.0.3",
                "info": {
                    "title": api_version.title,
                    "description": api_version.description,
                    "version": api_version.version,
                },
                "servers": api_version.servers,
                "paths": {},
                "components": {
                    "schemas": {},
                    "securitySchemes": api_version.security_schemes,
                },
            }

            # Add endpoints
            for endpoint in api_version.endpoints:
                if endpoint.path not in spec["paths"]:
                    spec["paths"][endpoint.path] = {}

                spec["paths"][endpoint.path][endpoint.method.lower()] = {
                    "summary": endpoint.summary,
                    "description": endpoint.description,
                    "parameters": endpoint.parameters,
                    "responses": endpoint.responses,
                    "tags": endpoint.tags,
                    "deprecated": endpoint.deprecated,
                }

                if endpoint.request_body:
                    spec["paths"][endpoint.path][endpoint.method.lower()][
                        "requestBody"
                    ] = endpoint.request_body

                if endpoint.security:
                    spec["paths"][endpoint.path][endpoint.method.lower()][
                        "security"
                    ] = endpoint.security

            return spec

        except Exception as e:
            logger.error("Generate OpenAPI spec error: %s", e)
            return {}

    def save_documentation(self, version: str, fmt: str = "json") -> bool:
        """Save API documentation to file."""
        try:
            spec = self.generate_openapi_spec(version)
            if not spec:
                return False

            file_path = self.docs_path / f"openapi_{version}.{fmt}"

            with open(file_path, "w", encoding="utf-8") as f:
                if fmt == "json":
                    json.dump(spec, f, indent=2)
                elif fmt == "yaml":
                    yaml.safe_dump(spec, f, sort_keys=False)
                else:
                    return False

            return True

        except Exception as e:
            logger.error("Save documentation error: %s", e)
            return False

    def load_documentation(
        self, version: str, fmt: str = "json"
    ) -> Dict[str, Any]:
        """Load API documentation from file."""
        try:
            file_path = self.docs_path / f"openapi_{version}.{fmt}"

            if not file_path.exists():
                return {}

            with open(file_path, "r", encoding="utf-8") as f:
                if fmt == "json":
                    return json.load(f)
                elif fmt == "yaml":
                    return yaml.safe_load(f)
                else:
                    return {}

        except Exception as e:
            logger.error("Load documentation error: %s", e)
            return {}

    def generate_from_routes(self, version: str, routes: List[Any]) -> bool:
        """Generate documentation from routes."""
        try:
            if version not in self.versions:
                return False

            for route in routes:
                methods = list(getattr(route, "methods", []))
                method = sorted(methods)[0] if methods else "GET"

                handler = getattr(route, "handler", None)
                summary = handler.__doc__ if handler and handler.__doc__ else ""
                description = inspect.getdoc(handler) if handler else None

                endpoint = APIEndpoint(
                    path=getattr(route, "path", "/unknown"),
                    method=method,
                    summary=summary,
                    description=description,
                    parameters=[],
                    responses={"200": {"description": "Successful response"}},
                )

                # Extract parameters from path
                param_convertors = getattr(route, "param_convertors", {}) or {}
                for param in param_convertors.keys():
                    endpoint.parameters.append(
                        {
                            "name": param,
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    )

                self.add_endpoint(version, endpoint)

            return True

        except Exception as e:
            logger.error("Generate from routes error: %s", e)
            return False

    def get_version_info(self, version: str) -> Dict[str, Any]:
        """Get API version information."""
        try:
            if version not in self.versions:
                return {}

            api_version = self.versions[version]

            return {
                "version": api_version.version,
                "title": api_version.title,
                "description": api_version.description,
                "endpoint_count": len(api_version.endpoints),
                "created_at": api_version.created_at.isoformat(),
            }

        except Exception as e:
            logger.error("Get version info error: %s", e)
            return {}

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all API versions."""
        try:
            return [
                self.get_version_info(version)
                for version in self.versions.keys()
            ]

        except Exception as e:
            logger.error("List versions error: %s", e)
            return []

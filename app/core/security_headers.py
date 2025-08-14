"""
Security headers module.
Manages security headers for responses.
"""

from typing import Any, Dict

from fastapi import Response

from app.core.config import get_settings

settings = get_settings()


class SecurityHeaders:
    """Security headers manager."""

    def __init__(self):
        """Initialize security headers."""
        self._default_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            "Content-Security-Policy": self._get_csp_policy(),
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": self._get_permissions_policy(),
            "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Permitted-Cross-Domain-Policies": "none",
            "Cross-Origin-Embedder-Policy": "require-corp",
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Resource-Policy": "same-origin",
            "X-DNS-Prefetch-Control": "off",
        }

    def _get_csp_policy(self) -> str:
        """Get Content Security Policy."""
        return (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "form-action 'self'; "
            "base-uri 'self'; "
            "object-src 'none'; "
            "media-src 'self'; "
            "worker-src 'self'; "
            "child-src 'self'; "
            "frame-src 'self'; "
            "manifest-src 'self'"
        )

    def _get_permissions_policy(self) -> str:
        """Get Permissions Policy."""
        return (
            "accelerometer=(), "
            "camera=(), "
            "geolocation=(), "
            "gyroscope=(), "
            "magnetometer=(), "
            "microphone=(), "
            "payment=(), "
            "usb=(), "
            "interest-cohort=()"
        )

    def add_headers(self, response: Response) -> None:
        """Add security headers to response."""
        response.headers.update(self._default_headers)

    def get_headers(self) -> Dict[str, str]:
        """Get security headers."""
        return self._default_headers.copy()

    def update_headers(self, headers: Dict[str, str]) -> None:
        """Update security headers."""
        self._default_headers.update(headers)

    def remove_header(self, header: str) -> None:
        """Remove security header."""
        self._default_headers.pop(header, None)

    def get_custom_headers(self, **kwargs: Any) -> Dict[str, str]:
        """Get custom security headers."""
        headers = self._default_headers.copy()
        headers.update(kwargs)
        return headers


security_headers = SecurityHeaders()

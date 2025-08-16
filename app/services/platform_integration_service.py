"""
Platform integration service.
Provides integration with various platforms and webhook support.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import aiohttp

from app.core.config import settings
from app.core.error_handling import (
    ErrorCategory,
    ErrorSeverity,
    ReputationError,
)
from app.core.metrics import track_performance
from app.core.constants import CONTENT_TYPE_JSON

class PlatformIntegrationService:
    """Service for platform integration and webhook support."""

    def __init__(self):
        """Initialize platform integration service."""
        self.platform_apis = {
            "glassdoor": {
                "base_url": settings.GLASSDOOR_API_URL,
                "api_key": settings.GLASSDOOR_API_KEY,
            },
            "quora": {
                "base_url": settings.QUORA_API_URL,
                "api_key": settings.QUORA_API_KEY,
            },
            "linkedin": {
                "base_url": settings.LINKEDIN_API_URL,
                "api_key": settings.LINKEDIN_API_KEY,
            },
            "app_store": {
                "base_url": settings.APP_STORE_API_URL,
                "api_key": settings.APP_STORE_API_KEY,
            },
            "play_store": {
                "base_url": settings.PLAY_STORE_API_URL,
                "api_key": settings.PLAY_STORE_API_KEY,
            },
            "telegram": {
                "base_url": settings.TELEGRAM_API_URL,
                "api_key": settings.TELEGRAM_API_KEY,
            },
            "discord": {
                "base_url": settings.DISCORD_API_URL,
                "api_key": settings.DISCORD_API_KEY,
            },
        }

        self.webhook_clients = {}

    @track_performance
    async def fetch_platform_data(
        self,
        platform: str,
        query: Dict[str, Any],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch data from a specific platform."""
        try:
            if platform not in self.platform_apis:
                raise ReputationError(
                    message=f"Unsupported platform: {platform}",
                    severity=ErrorSeverity.MEDIUM,
                    category=ErrorCategory.BUSINESS,
                )

            platform_config = self.platform_apis[platform]

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                headers = {
                    "Authorization": f"Bearer {platform_config['api_key']}",
                    "Content-Type": CONTENT_TYPE_JSON,
                }

                params = {
                    **query,
                    "start_date": start_date.isoformat()
                    if start_date
                    else None,
                    "end_date": end_date.isoformat() if end_date else None,
                }

                async with session.get(
                    platform_config["base_url"], headers=headers, params=params
                ) as response:
                    if response.status != 200:
                        raise ReputationError(
                            message=f"Error fetching data from {platform}: {await response.text()}",
                            severity=ErrorSeverity.HIGH,
                            category=ErrorCategory.BUSINESS,
                        )

                    return await response.json()
        except Exception as e:
            raise ReputationError(
                message=f"Error in platform data fetch: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS,
            )

    @track_performance
    async def register_webhook(
        self,
        client_id: str,
        webhook_url: str,
        events: List[str],
        secret: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register a new webhook client."""
        try:
            self.webhook_clients[client_id] = {
                "url": webhook_url,
                "events": events,
                "secret": secret,
                "created_at": datetime.now(timezone.utc),
            }

            return {
                "client_id": client_id,
                "status": "registered",
                "events": events,
            }
        except Exception as e:
            raise ReputationError(
                message=f"Error registering webhook: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS,
            )

    @track_performance
    async def send_webhook_notification(
        self, client_id: str, event: str, data: Dict[str, Any]
    ) -> bool:
        """Send webhook notification to a client."""
        try:
            if client_id not in self.webhook_clients:
                raise ReputationError(
                    message=f"Webhook client not found: {client_id}",
                    severity=ErrorSeverity.MEDIUM,
                    category=ErrorCategory.BUSINESS,
                )

            client = self.webhook_clients[client_id]

            if event not in client["events"]:
                raise ReputationError(
                    message=f"Event not subscribed: {event}",
                    severity=ErrorSeverity.MEDIUM,
                    category=ErrorCategory.BUSINESS,
                )

            payload = {
                "event": event,
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            headers = {"Content-Type": CONTENT_TYPE_JSON}

            if client["secret"]:
                # Add signature if secret is provided
                headers["X-Webhook-Signature"] = self._generate_signature(
                    payload, client["secret"]
                )

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(
                    client["url"], json=payload, headers=headers
                ) as response:
                    if response.status != 200:
                        raise ReputationError(
                            message=f"Error sending webhook: {await response.text()}",
                            severity=ErrorSeverity.HIGH,
                            category=ErrorCategory.BUSINESS,
                        )

                    return True
        except Exception as e:
            raise ReputationError(
                message=f"Error in webhook notification: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS,
            )

    def _generate_signature(self, payload: Dict[str, Any], secret: str) -> str:
        """Generate webhook signature."""
        import hashlib
        import hmac
        import json

        message = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            secret.encode(), message.encode(), hashlib.sha256
        ).hexdigest()

        return signature

    @track_performance
    async def fetch_dark_web_mentions(
        self, query: Dict[str, Any], max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetch mentions from dark web sources."""
        try:
            # This is a placeholder for dark web scraping implementation
            # In a real implementation, this would use secure scraping techniques
            # and proper authentication/authorization

            return []
        except Exception as e:
            raise ReputationError(
                message=f"Error in dark web mentions fetch: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS,
            )

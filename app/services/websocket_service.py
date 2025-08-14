"""
WebSocket service for real-time communication.
Provides WebSocket connection and message handling.
"""

import asyncio
import json
import logging
from asyncio import Lock, Semaphore
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Constants
MAX_MESSAGE_SIZE = 1024 * 1024  # 1MB
CONNECTION_TIMEOUT = 300  # 5 minutes
MAX_CONCURRENT_OPERATIONS = 100


class WebSocketConnection:
    """WebSocket connection representation."""

    def __init__(
        self,
        connection_id: str,
        send: Callable,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize connection."""
        self.connection_id = connection_id
        self.send = send
        self.metadata = metadata or {}
        self.subscriptions: Set[str] = set()
        self.connected_at = datetime.utcnow()
        self.last_message = datetime.utcnow()
        self.lock = Lock()

    def is_stale(self) -> bool:
        """Check if connection is stale."""
        return (datetime.utcnow() - self.last_message) > timedelta(
            seconds=CONNECTION_TIMEOUT
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert connection to dictionary."""
        return {
            "connection_id": self.connection_id,
            "metadata": self.metadata,
            "subscriptions": list(self.subscriptions),
            "connected_at": self.connected_at.isoformat(),
            "last_message": self.last_message.isoformat(),
        }


class WebSocketChannel:
    """WebSocket channel representation."""

    def __init__(
        self, channel_id: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize channel."""
        self.channel_id = channel_id
        self.metadata = metadata or {}
        self.connections: Dict[str, WebSocketConnection] = {}
        self.message_handlers: List[Callable] = []
        self.created_at = datetime.utcnow()
        self.lock = Lock()
        self.semaphore = Semaphore(MAX_CONCURRENT_OPERATIONS)

    async def add_connection(self, connection: WebSocketConnection) -> bool:
        """Add connection to channel."""
        try:
            async with self.lock:
                if connection.connection_id in self.connections:
                    return False

                self.connections[connection.connection_id] = connection
                return True

        except Exception as e:
            logger.error(f"Add connection error: {str(e)}")
            return False

    async def remove_connection(self, connection_id: str) -> bool:
        """Remove connection from channel."""
        try:
            async with self.lock:
                if connection_id not in self.connections:
                    return False

                del self.connections[connection_id]
                return True

        except Exception as e:
            logger.error(f"Remove connection error: {str(e)}")
            return False

    def add_message_handler(self, handler: Callable):
        """Add message handler."""
        self.message_handlers.append(handler)

    async def broadcast(
        self, message: Dict[str, Any], exclude: Optional[List[str]] = None
    ):
        """Broadcast message to all connections."""
        try:
            if not self._validate_message(message):
                logger.error("Invalid message format")
                return

            exclude_set = set(exclude) if exclude else set()

            async with self.semaphore:
                for connection in list(self.connections.values()):
                    if connection.connection_id not in exclude_set:
                        if not connection.is_stale():
                            await connection.send(message)
                        else:
                            await self.remove_connection(
                                connection.connection_id
                            )

        except Exception as e:
            logger.error(f"Broadcast error: {str(e)}")

    def _validate_message(self, message: Dict[str, Any]) -> bool:
        """Validate message format and size."""
        try:
            # Check message size
            message_str = json.dumps(message)
            if len(message_str.encode("utf-8")) > MAX_MESSAGE_SIZE:
                return False

            # Basic message structure validation
            if not isinstance(message, dict):
                return False

            return True

        except Exception:
            return False

    async def handle_message(
        self, connection_id: str, message: Dict[str, Any]
    ):
        """Handle incoming message."""
        try:
            if not self._validate_message(message):
                logger.error("Invalid message format")
                return

            connection = self.connections.get(connection_id)
            if not connection:
                return

            connection.last_message = datetime.utcnow()

            # Trigger handlers
            async with self.semaphore:
                for handler in self.message_handlers:
                    try:
                        await handler(connection, message)
                    except Exception as e:
                        logger.error(f"Handler error: {str(e)}")

        except Exception as e:
            logger.error(f"Handle message error: {str(e)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert channel to dictionary."""
        return {
            "channel_id": self.channel_id,
            "metadata": self.metadata,
            "connections": len(self.connections),
            "created_at": self.created_at.isoformat(),
        }


class WebSocketService:
    """WebSocket management service."""

    def __init__(self):
        """Initialize WebSocket service."""
        self.channels: Dict[str, WebSocketChannel] = {}
        self.connections: Dict[str, WebSocketConnection] = {}
        self.lock = Lock()
        self.cleanup_task = None

    async def start(self):
        """Start the WebSocket service."""
        self.cleanup_task = asyncio.create_task(
            self._cleanup_stale_connections()
        )

    async def stop(self):
        """Stop the WebSocket service."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_stale_connections(self):
        """Periodically clean up stale connections."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                async with self.lock:
                    for connection_id, connection in list(
                        self.connections.items()
                    ):
                        if connection.is_stale():
                            await self.unregister_connection(connection_id)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {str(e)}")

    async def create_channel(
        self,
        channel_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create new channel."""
        try:
            channel_id = channel_id or str(uuid4())

            async with self.lock:
                if channel_id in self.channels:
                    return {
                        "status": "error",
                        "message": "Channel already exists",
                    }

                channel = WebSocketChannel(channel_id, metadata)
                self.channels[channel_id] = channel

                return {
                    "status": "success",
                    "message": "Channel created successfully",
                    "channel": channel.to_dict(),
                }

        except Exception as e:
            logger.error(f"Create channel error: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def delete_channel(self, channel_id: str) -> Dict[str, Any]:
        """Delete channel."""
        try:
            async with self.lock:
                if channel_id not in self.channels:
                    return {"status": "error", "message": "Channel not found"}

                channel = self.channels[channel_id]

                # Remove all connections
                for connection in list(channel.connections.values()):
                    connection.subscriptions.remove(channel_id)

                del self.channels[channel_id]

                return {
                    "status": "success",
                    "message": "Channel deleted successfully",
                }

        except Exception as e:
            logger.error(f"Delete channel error: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def register_connection(
        self, send: Callable, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Register new connection."""
        try:
            connection_id = str(uuid4())
            connection = WebSocketConnection(connection_id, send, metadata)

            async with self.lock:
                self.connections[connection_id] = connection

            return {
                "status": "success",
                "message": "Connection registered successfully",
                "connection": connection.to_dict(),
            }

        except Exception as e:
            logger.error(f"Register connection error: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def unregister_connection(
        self, connection_id: str
    ) -> Dict[str, Any]:
        """Unregister connection."""
        try:
            async with self.lock:
                if connection_id not in self.connections:
                    return {
                        "status": "error",
                        "message": "Connection not found",
                    }

                connection = self.connections[connection_id]

                # Remove from all channels
                for channel_id in list(connection.subscriptions):
                    channel = self.channels.get(channel_id)
                    if channel:
                        await channel.remove_connection(connection_id)

                del self.connections[connection_id]

                return {
                    "status": "success",
                    "message": "Connection unregistered successfully",
                }

        except Exception as e:
            logger.error(f"Unregister connection error: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def subscribe(
        self, connection_id: str, channel_id: str
    ) -> Dict[str, Any]:
        """Subscribe connection to channel."""
        try:
            async with self.lock:
                connection = self.connections.get(connection_id)
                if not connection:
                    return {
                        "status": "error",
                        "message": "Connection not found",
                    }

                channel = self.channels.get(channel_id)
                if not channel:
                    return {"status": "error", "message": "Channel not found"}

                connection.subscriptions.add(channel_id)
                await channel.add_connection(connection)

                return {
                    "status": "success",
                    "message": "Subscribed successfully",
                }

        except Exception as e:
            logger.error(f"Subscribe error: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def unsubscribe(
        self, connection_id: str, channel_id: str
    ) -> Dict[str, Any]:
        """Unsubscribe connection from channel."""
        try:
            async with self.lock:
                connection = self.connections.get(connection_id)
                if not connection:
                    return {
                        "status": "error",
                        "message": "Connection not found",
                    }

                channel = self.channels.get(channel_id)
                if not channel:
                    return {"status": "error", "message": "Channel not found"}

                connection.subscriptions.remove(channel_id)
                await channel.remove_connection(connection_id)

                return {
                    "status": "success",
                    "message": "Unsubscribed successfully",
                }

        except Exception as e:
            logger.error(f"Unsubscribe error: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def broadcast(
        self,
        channel_id: str,
        message: Dict[str, Any],
        exclude: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Broadcast message to channel."""
        try:
            async with self.lock:
                channel = self.channels.get(channel_id)
                if not channel:
                    return {"status": "error", "message": "Channel not found"}

                await channel.broadcast(message, exclude)

                return {
                    "status": "success",
                    "message": "Message broadcasted successfully",
                }

        except Exception as e:
            logger.error(f"Broadcast error: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def handle_message(
        self, connection_id: str, message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle incoming message."""
        try:
            async with self.lock:
                connection = self.connections.get(connection_id)
                if not connection:
                    return {
                        "status": "error",
                        "message": "Connection not found",
                    }

                # Handle message in all subscribed channels
                for channel_id in list(connection.subscriptions):
                    channel = self.channels.get(channel_id)
                    if channel:
                        await channel.handle_message(connection_id, message)

                return {
                    "status": "success",
                    "message": "Message handled successfully",
                }

        except Exception as e:
            logger.error(f"Handle message error: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def get_channel_info(self, channel_id: str) -> Dict[str, Any]:
        """Get channel information."""
        try:
            async with self.lock:
                channel = self.channels.get(channel_id)
                if not channel:
                    return {"status": "error", "message": "Channel not found"}

                connections = [
                    connection.to_dict()
                    for connection in channel.connections.values()
                ]

                return {
                    "status": "success",
                    "channel": channel.to_dict(),
                    "connections": connections,
                }

        except Exception as e:
            logger.error(f"Get channel info error: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def get_connection_info(self, connection_id: str) -> Dict[str, Any]:
        """Get connection information."""
        try:
            async with self.lock:
                connection = self.connections.get(connection_id)
                if not connection:
                    return {
                        "status": "error",
                        "message": "Connection not found",
                    }

                subscribed_channels = [
                    self.channels[channel_id].to_dict()
                    for channel_id in connection.subscriptions
                    if channel_id in self.channels
                ]

                return {
                    "status": "success",
                    "connection": connection.to_dict(),
                    "channels": subscribed_channels,
                }

        except Exception as e:
            logger.error(f"Get connection info error: {str(e)}")
            return {"status": "error", "message": str(e)}

"""
Real-time notification service.
Provides instant alerts for reputation changes and important events.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis.asyncio as redis

from app.core.config import get_settings
from app.core.metrics import NOTIFICATION_LATENCY
from app.core.optimizations import CircuitBreaker

logger = logging.getLogger(__name__)
settings = get_settings()


class NotificationChannel:
    """Base class for notification channels."""

    async def send_notification(
        self, user_id: str, message: str, data: Dict[str, Any]
    ) -> bool:
        """Send notification through channel."""
        raise NotImplementedError


class EmailNotifier(NotificationChannel):
    """Email notification channel."""

    async def send_notification(
        self, user_id: str, message: str, data: Dict[str, Any]
    ) -> bool:
        """Send email notification."""
        try:
            # Implement email sending logic
            return True
        except Exception as e:
            logger.error(f"Email notification error: {str(e)}")
            return False


class WebhookNotifier(NotificationChannel):
    """Webhook notification channel."""

    async def send_notification(
        self, user_id: str, message: str, data: Dict[str, Any]
    ) -> bool:
        """Send webhook notification."""
        try:
            # Implement webhook sending logic
            return True
        except Exception as e:
            logger.error(f"Webhook notification error: {str(e)}")
            return False


class WebSocketNotifier(NotificationChannel):
    """WebSocket notification channel."""

    def __init__(self):
        """Initialize WebSocket notifier."""
        self.connections = {}
        self.redis = redis.from_url(settings.REDIS_URL)

    async def register_connection(self, user_id: str, websocket: Any):
        """Register WebSocket connection."""
        self.connections[user_id] = websocket

    async def remove_connection(self, user_id: str):
        """Remove WebSocket connection."""
        if user_id in self.connections:
            del self.connections[user_id]

    async def send_notification(
        self, user_id: str, message: str, data: Dict[str, Any]
    ) -> bool:
        """Send WebSocket notification."""
        try:
            if user_id in self.connections:
                websocket = self.connections[user_id]
                await websocket.send_json(
                    {
                        "message": message,
                        "data": data,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
                return True
            return False
        except Exception as e:
            logger.error(f"WebSocket notification error: {str(e)}")
            return False


class SlackNotifier(NotificationChannel):
    """Slack notification channel."""

    async def send_notification(
        self, user_id: str, message: str, data: Dict[str, Any]
    ) -> bool:
        """Send Slack notification."""
        try:
            # Implement Slack sending logic
            return True
        except Exception as e:
            logger.error(f"Slack notification error: {str(e)}")
            return False


class NotificationBatcher:
    """Notification batching system."""

    def __init__(self):
        """Initialize notification batcher."""
        self.redis = redis.from_url(settings.REDIS_URL)
        self.batch_size = 10
        self.batch_window = 300  # 5 minutes

    async def add_to_batch(self, user_id: str, notification: Dict[str, Any]):
        """Add notification to batch."""
        batch_key = f"notification:batch:{user_id}"

        # Add to batch
        await self.redis.lpush(batch_key, json.dumps(notification))

        # Set expiry if first item
        if await self.redis.llen(batch_key) == 1:
            await self.redis.expire(batch_key, self.batch_window)

    async def get_batch(self, user_id: str) -> List[Dict[str, Any]]:
        """Get current batch of notifications."""
        batch_key = f"notification:batch:{user_id}"
        notifications = await self.redis.lrange(batch_key, 0, -1)

        return [json.loads(notification) for notification in notifications]

    async def should_send_batch(self, user_id: str) -> bool:
        """Check if batch should be sent."""
        batch_key = f"notification:batch:{user_id}"
        batch_size = await self.redis.llen(batch_key)

        return batch_size >= self.batch_size


class PriorityQueue:
    """Priority-based notification queue."""

    def __init__(self):
        """Initialize priority queue."""
        self.redis = redis.from_url(settings.REDIS_URL)
        self.priorities = {"critical": 3, "high": 2, "medium": 1, "low": 0}

    async def enqueue(
        self, notification: Dict[str, Any], priority: str = "medium"
    ):
        """Add notification to priority queue."""
        score = self.priorities.get(priority, 1)

        await self.redis.zadd(
            "notification:queue", {json.dumps(notification): score}
        )

    async def dequeue(self) -> Optional[Dict[str, Any]]:
        """Get highest priority notification."""
        # Get highest scored item
        items = await self.redis.zrevrange(
            "notification:queue", 0, 0, withscores=True
        )

        if items:
            notification_json, score = items[0]

            # Remove from queue
            await self.redis.zrem("notification:queue", notification_json)

            return json.loads(notification_json)

        return None

    async def get_queue_size(self) -> Dict[str, int]:
        """Get size of queue by priority."""
        sizes = {}

        for priority, score in self.priorities.items():
            count = await self.redis.zcount("notification:queue", score, score)
            sizes[priority] = count

        return sizes


class DeliveryTracker:
    """Notification delivery tracking system."""

    def __init__(self):
        """Initialize delivery tracker."""
        self.redis = redis.from_url(settings.REDIS_URL)

    async def track_delivery(
        self, notification_id: str, user_id: str, channel: str, success: bool
    ):
        """Track notification delivery attempt."""
        record = {
            "notification_id": notification_id,
            "user_id": user_id,
            "channel": channel,
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Store delivery record
        await self.redis.hset(
            f"notification:delivery:{notification_id}",
            channel,
            json.dumps(record),
        )

        # Update success rates
        await self._update_success_rates(channel, success)

    async def get_delivery_status(
        self, notification_id: str
    ) -> Dict[str, Any]:
        """Get delivery status for notification."""
        deliveries = await self.redis.hgetall(
            f"notification:delivery:{notification_id}"
        )

        return {
            channel: json.loads(status)
            for channel, status in deliveries.items()
        }

    async def _update_success_rates(self, channel: str, success: bool):
        """Update channel success rates."""
        key = f"notification:stats:{channel}"

        async with self.redis.pipeline() as pipe:
            await pipe.hincrby(key, "total", 1)
            if success:
                await pipe.hincrby(key, "successful", 1)
            await pipe.execute()


class NotificationService:
    """Real-time notification service."""

    def __init__(self):
        """Initialize notification service."""
        self.channels = {
            "email": EmailNotifier(),
            "webhook": WebhookNotifier(),
            "websocket": WebSocketNotifier(),
            "slack": SlackNotifier(),
        }
        self.redis = redis.from_url(settings.REDIS_URL)
        self.batcher = NotificationBatcher()
        self.queue = PriorityQueue()
        self.tracker = DeliveryTracker()

    @CircuitBreaker(failure_threshold=3, reset_timeout=30)
    async def send_notification(
        self,
        user_id: str,
        notification_type: str,
        message: str,
        data: Dict[str, Any],
        channels: Optional[List[str]] = None,
        priority: str = "medium",
        batch: bool = False,
    ) -> Dict[str, Any]:
        """Send notification through specified channels."""
        start_time = datetime.utcnow()

        notification = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "type": notification_type,
            "message": message,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if batch:
            # Add to batch
            await self.batcher.add_to_batch(user_id, notification)

            # Check if batch should be sent
            if await self.batcher.should_send_batch(user_id):
                return await self._send_batch(user_id)

            return {"status": "batched"}

        # Add to priority queue
        await self.queue.enqueue(notification, priority)

        # Process queue
        results = await self._process_queue()

        # Record latency
        duration = (datetime.utcnow() - start_time).total_seconds()
        NOTIFICATION_LATENCY.observe(duration)

        return results

    async def _send_batch(self, user_id: str) -> Dict[str, Any]:
        """Send batch of notifications."""
        notifications = await self.batcher.get_batch(user_id)

        if not notifications:
            return {"status": "no_notifications"}

        results = {}
        channels = await self._get_user_channels(user_id)

        for notification in notifications:
            for channel in channels:
                if channel in self.channels:
                    success = await self.channels[channel].send_notification(
                        user_id,
                        self._format_batch_message(notifications),
                        {"notifications": notifications},
                    )
                    results[channel] = success

                    # Track delivery
                    await self.tracker.track_delivery(
                        notification["id"], user_id, channel, success
                    )

        return {
            "status": "sent",
            "batch_size": len(notifications),
            "results": results,
        }

    def _format_batch_message(
        self, notifications: List[Dict[str, Any]]
    ) -> str:
        """Format batch notification message."""
        return f"You have {len(notifications)} new notifications"

    async def _process_queue(self) -> Dict[str, Any]:
        """Process notification queue."""
        notification = await self.queue.dequeue()

        if not notification:
            return {"status": "no_notifications"}

        channels = await self._get_user_channels(notification["user_id"])
        results = {}

        for channel in channels:
            if channel in self.channels:
                success = await self.channels[channel].send_notification(
                    notification["user_id"],
                    notification["message"],
                    notification["data"],
                )
                results[channel] = success

                # Track delivery
                await self.tracker.track_delivery(
                    notification["id"],
                    notification["user_id"],
                    channel,
                    success,
                )

        return {
            "status": "sent",
            "notification_id": notification["id"],
            "results": results,
        }

    async def get_delivery_status(
        self, notification_id: str
    ) -> Dict[str, Any]:
        """Get notification delivery status."""
        return await self.tracker.get_delivery_status(notification_id)

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get notification queue status."""
        return {
            "queue_size": await self.queue.get_queue_size(),
            "active_batches": await self._get_active_batch_count(),
        }

    async def _get_active_batch_count(self) -> int:
        """Get count of active notification batches."""
        keys = await self.redis.keys("notification:batch:*")
        return len(keys)

    async def _get_user_channels(self, user_id: str) -> List[str]:
        """Get user's preferred notification channels."""
        channels = await self.redis.get(f"user:channels:{user_id}")
        if channels:
            return channels.split(",")
        return ["email"]  # Default to email

    async def get_user_notifications(
        self, user_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get user's notification history."""
        notifications = await self.redis.lrange(
            f"user:notifications:{user_id}", 0, limit - 1
        )

        return [json.loads(notification) for notification in notifications]

    async def mark_notification_read(
        self, user_id: str, notification_id: str
    ) -> bool:
        """Mark notification as read."""
        key = f"user:notifications:read:{user_id}"
        return await self.redis.sadd(key, notification_id) > 0

    async def get_unread_count(self, user_id: str) -> int:
        """Get count of unread notifications."""
        all_notifications = await self.get_user_notifications(user_id)
        read_notifications = await self.redis.smembers(
            f"user:notifications:read:{user_id}"
        )

        return len(
            [n for n in all_notifications if n["id"] not in read_notifications]
        )

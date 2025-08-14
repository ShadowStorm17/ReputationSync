"""
Event sourcing service for event storage and replay.
Provides event sourcing and event store functionality.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EventType(str, Enum):
    """Event types."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    CUSTOM = "custom"


class Event(BaseModel):
    """Event representation."""

    event_id: str
    aggregate_id: str
    aggregate_type: str
    event_type: EventType
    event_data: Dict[str, Any]
    metadata: Dict[str, Any]
    version: int
    timestamp: datetime = datetime.utcnow()


class EventStore:
    """Event store implementation."""

    def __init__(self):
        """Initialize event store."""
        self.events: Dict[str, List[Event]] = {}
        self.snapshots: Dict[str, Dict[str, Any]] = {}
        self.snapshot_frequency = 100
        self.handlers: Dict[str, List[Callable]] = {}

    async def append_event(self, event: Event) -> bool:
        """Append event to store."""
        try:
            aggregate_key = f"{event.aggregate_type}:{event.aggregate_id}"

            if aggregate_key not in self.events:
                self.events[aggregate_key] = []

            # Verify version
            current_version = len(self.events[aggregate_key])
            if event.version != current_version + 1:
                return False

            # Store event
            self.events[aggregate_key].append(event)

            # Create snapshot if needed
            if len(self.events[aggregate_key]) % self.snapshot_frequency == 0:
                await self.create_snapshot(
                    event.aggregate_type, event.aggregate_id
                )

            # Trigger handlers
            await self.trigger_handlers(event)

            return True

        except Exception as e:
            logger.error(f"Append event error: {str(e)}")
            return False

    async def get_events(
        self,
        aggregate_type: str,
        aggregate_id: str,
        start_version: Optional[int] = None,
        end_version: Optional[int] = None,
    ) -> List[Event]:
        """Get events for aggregate."""
        try:
            aggregate_key = f"{aggregate_type}:{aggregate_id}"

            if aggregate_key not in self.events:
                return []

            events = self.events[aggregate_key]

            if start_version is not None:
                events = [e for e in events if e.version >= start_version]

            if end_version is not None:
                events = [e for e in events if e.version <= end_version]

            return events

        except Exception as e:
            logger.error(f"Get events error: {str(e)}")
            return []

    async def create_snapshot(
        self, aggregate_type: str, aggregate_id: str
    ) -> bool:
        """Create aggregate snapshot."""
        try:
            aggregate_key = f"{aggregate_type}:{aggregate_id}"

            if aggregate_key not in self.events:
                return False

            # Get all events
            events = self.events[aggregate_key]

            # Apply events to create snapshot
            snapshot = {}
            for event in events:
                if event.event_type == EventType.CREATE:
                    snapshot.update(event.event_data)
                elif event.event_type == EventType.UPDATE:
                    snapshot.update(event.event_data)
                elif event.event_type == EventType.DELETE:
                    snapshot = {}
                elif event.event_type == EventType.CUSTOM:
                    # Custom events need specific handling
                    pass

            self.snapshots[aggregate_key] = {
                "data": snapshot,
                "version": len(events),
                "timestamp": datetime.utcnow().isoformat(),
            }

            return True

        except Exception as e:
            logger.error(f"Create snapshot error: {str(e)}")
            return False

    async def get_snapshot(
        self, aggregate_type: str, aggregate_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get aggregate snapshot."""
        try:
            aggregate_key = f"{aggregate_type}:{aggregate_id}"
            return self.snapshots.get(aggregate_key)

        except Exception as e:
            logger.error(f"Get snapshot error: {str(e)}")
            return None

    def register_handler(self, event_type: str, handler: Callable):
        """Register event handler."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []

        self.handlers[event_type].append(handler)

    async def trigger_handlers(self, event: Event):
        """Trigger event handlers."""
        try:
            handlers = self.handlers.get(event.event_type, [])

            for handler in handlers:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Handler error: {str(e)}")

        except Exception as e:
            logger.error(f"Trigger handlers error: {str(e)}")


class EventSourcingService:
    """Event sourcing management service."""

    def __init__(self):
        """Initialize event sourcing service."""
        self.store = EventStore()

    async def create_aggregate(
        self,
        aggregate_type: str,
        aggregate_id: Optional[str] = None,
        data: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Create new aggregate."""
        try:
            aggregate_id = aggregate_id or str(uuid4())

            event = Event(
                event_id=str(uuid4()),
                aggregate_id=aggregate_id,
                aggregate_type=aggregate_type,
                event_type=EventType.CREATE,
                event_data=data or {},
                metadata={},
                version=1,
            )

            success = await self.store.append_event(event)

            return {
                "status": "success" if success else "error",
                "message": (
                    "Aggregate created successfully"
                    if success
                    else "Failed to create aggregate"
                ),
                "aggregate_id": aggregate_id if success else None,
            }

        except Exception as e:
            logger.error(f"Create aggregate error: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def update_aggregate(
        self,
        aggregate_type: str,
        aggregate_id: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update aggregate."""
        try:
            # Get current version
            events = await self.store.get_events(aggregate_type, aggregate_id)

            if not events:
                return {"status": "error", "message": "Aggregate not found"}

            event = Event(
                event_id=str(uuid4()),
                aggregate_id=aggregate_id,
                aggregate_type=aggregate_type,
                event_type=EventType.UPDATE,
                event_data=data,
                metadata=metadata or {},
                version=len(events) + 1,
            )

            success = await self.store.append_event(event)

            return {
                "status": "success" if success else "error",
                "message": (
                    "Aggregate updated successfully"
                    if success
                    else "Failed to update aggregate"
                ),
            }

        except Exception as e:
            logger.error(f"Update aggregate error: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def delete_aggregate(
        self,
        aggregate_type: str,
        aggregate_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Delete aggregate."""
        try:
            # Get current version
            events = await self.store.get_events(aggregate_type, aggregate_id)

            if not events:
                return {"status": "error", "message": "Aggregate not found"}

            event = Event(
                event_id=str(uuid4()),
                aggregate_id=aggregate_id,
                aggregate_type=aggregate_type,
                event_type=EventType.DELETE,
                event_data={},
                metadata=metadata or {},
                version=len(events) + 1,
            )

            success = await self.store.append_event(event)

            return {
                "status": "success" if success else "error",
                "message": (
                    "Aggregate deleted successfully"
                    if success
                    else "Failed to delete aggregate"
                ),
            }

        except Exception as e:
            logger.error(f"Delete aggregate error: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def get_aggregate(
        self, aggregate_type: str, aggregate_id: str
    ) -> Dict[str, Any]:
        """Get aggregate state."""
        try:
            # Try to get snapshot first
            snapshot = await self.store.get_snapshot(
                aggregate_type, aggregate_id
            )

            if snapshot:
                # Get events after snapshot
                events = await self.store.get_events(
                    aggregate_type,
                    aggregate_id,
                    start_version=snapshot["version"] + 1,
                )

                # Apply events to snapshot
                state = snapshot["data"].copy()
                for event in events:
                    if event.event_type == EventType.UPDATE:
                        state.update(event.event_data)
                    elif event.event_type == EventType.DELETE:
                        state = {}

                return {
                    "status": "success",
                    "state": state,
                    "version": snapshot["version"] + len(events),
                }

            # No snapshot, get all events
            events = await self.store.get_events(aggregate_type, aggregate_id)

            if not events:
                return {"status": "error", "message": "Aggregate not found"}

            # Apply all events
            state = {}
            for event in events:
                if event.event_type == EventType.CREATE:
                    state.update(event.event_data)
                elif event.event_type == EventType.UPDATE:
                    state.update(event.event_data)
                elif event.event_type == EventType.DELETE:
                    state = {}

            return {
                "status": "success",
                "state": state,
                "version": len(events),
            }

        except Exception as e:
            logger.error(f"Get aggregate error: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def get_aggregate_history(
        self, aggregate_type: str, aggregate_id: str
    ) -> Dict[str, Any]:
        """Get aggregate event history."""
        try:
            events = await self.store.get_events(aggregate_type, aggregate_id)

            return {
                "status": "success",
                "events": [event.dict() for event in events],
            }

        except Exception as e:
            logger.error(f"Get history error: {str(e)}")
            return {"status": "error", "message": str(e)}

    def register_handler(self, event_type: str, handler: Callable):
        """Register event handler."""
        self.store.register_handler(event_type, handler)

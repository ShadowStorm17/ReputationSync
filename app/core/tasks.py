"""
Background task management module.
Handles asynchronous processing of heavy computations.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from app.core.cache import cache_manager

logger = logging.getLogger(__name__)


class TaskManager:
    """Manages background tasks and their execution."""

    def __init__(self):
        """Initialize task manager."""
        self.tasks: Dict[str, asyncio.Task] = {}
        self.results: Dict[str, Any] = {}

    async def execute_task(
        self, task_id: str, func: Callable, *args, **kwargs
    ) -> str:
        """Execute a task asynchronously."""
        try:
            # Create task
            task = asyncio.create_task(func(*args, **kwargs))
            self.tasks[task_id] = task

            # Store task metadata
            self.results[task_id] = {
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "args": args,
                "kwargs": kwargs,
            }

            # Add callback for task completion
            task.add_done_callback(
                lambda t: self._handle_task_completion(task_id, t)
            )

            return task_id

        except asyncio.CancelledError:
            logger.warning("Task %s was cancelled during execution", task_id)
            self.results[task_id] = {
                "status": "cancelled",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            raise
        except Exception as e:
            logger.error("Error executing task %s: %s", task_id, e)
            self.results[task_id] = {
                "status": "failed",
                "error": str(e),
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            raise

    def _handle_task_completion(self, task_id: str, task: asyncio.Task):
        """Handle task completion."""
        try:
            if task.cancelled():
                self.results[task_id]["status"] = "cancelled"
            elif task.exception():
                self.results[task_id].update(
                    {"status": "failed", "error": str(task.exception())}
                )
            else:
                result = task.result()
                self.results[task_id].update(
                    {"status": "completed", "result": result}
                )

                # Cache result if applicable
                if isinstance(result, dict) and "cache_key" in result:
                    asyncio.create_task(
                        cache_manager.set(
                            result["cache_key"],
                            result["data"],
                            expire=result.get("expire"),
                        )
                    )

        except asyncio.CancelledError:
            self.results[task_id]["status"] = "cancelled"
        except Exception as e:
            logger.error("Error handling task completion %s: %s", task_id, e)
            self.results[task_id].update({"status": "failed", "error": str(e)})

    async def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get task status and result."""
        return self.results.get(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if not task.done():
                task.cancel()
                return True
        return False


# Create global task manager instance
task_manager = TaskManager()

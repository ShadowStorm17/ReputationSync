"""
AI runner service.
Manages continuous AI processing and task execution.
"""

import asyncio
import logging
from typing import Dict

from app.core.ai_config import ai_config
from app.core.error_handling import ErrorCategory, ErrorSeverity, handle_errors

logger = logging.getLogger(__name__)


class AIRunner:
    """AI runner service."""

    def __init__(self):
        """Initialize AI runner."""
        self.config = ai_config.get_continuous_config()
        self.running = False
        self.tasks: Dict[str, asyncio.Task] = {}

    @handle_errors(severity=ErrorSeverity.HIGH, category=ErrorCategory.SYSTEM)
    async def start(self):
        """Start continuous processing."""
        if not self.running:
            logger.info("Starting AI runner...")
            self.running = True
            self.tasks["process"] = asyncio.create_task(
                self._process_continuously()
            )

    @handle_errors(severity=ErrorSeverity.HIGH, category=ErrorCategory.SYSTEM)
    async def stop(self):
        """Stop continuous processing."""
        if self.running:
            logger.info("Stopping AI runner...")
            self.running = False
            for task in self.tasks.values():
                task.cancel()
            self.tasks.clear()

    async def _process_continuously(self):
        """Run continuous processing."""
        while self.running:
            try:
                # Process AI tasks
                logger.info("Processing AI tasks...")
                await asyncio.sleep(self.config["interval"])

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in continuous processing: {str(e)}")
                await asyncio.sleep(10)  # Wait before retrying


# Global instance
ai_runner = AIRunner()

"""
Queue service for background task processing.
Provides job queuing and processing capabilities.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

import redis.asyncio as redis

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class Job:
    """Background job representation."""

    def __init__(
        self,
        job_id: str,
        task_name: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        priority: int = 0,
        retry_count: int = 3,
        retry_delay: int = 60,
    ):
        """Initialize job."""
        self.job_id = job_id
        self.task_name = task_name
        self.args = args
        self.kwargs = kwargs
        self.priority = priority
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.attempts = 0
        self.status = "pending"
        self.result = None
        self.error = None
        self.created_at = datetime.now(timezone.utc)
        self.started_at = None
        self.completed_at = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "job_id": self.job_id,
            "task_name": self.task_name,
            "args": self.args,
            "kwargs": self.kwargs,
            "priority": self.priority,
            "retry_count": self.retry_count,
            "retry_delay": self.retry_delay,
            "attempts": self.attempts,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat()
            if self.started_at
            else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        """Create job from dictionary."""
        job = cls(
            job_id=data["job_id"],
            task_name=data["task_name"],
            args=data["args"],
            kwargs=data["kwargs"],
            priority=data["priority"],
            retry_count=data["retry_count"],
            retry_delay=data["retry_delay"],
        )
        job.attempts = data["attempts"]
        job.status = data["status"]
        job.result = data["result"]
        job.error = data["error"]
        job.created_at = datetime.fromisoformat(data["created_at"])
        if data["started_at"]:
            job.started_at = datetime.fromisoformat(data["started_at"])
        if data["completed_at"]:
            job.completed_at = datetime.fromisoformat(data["completed_at"])
        return job


class Queue:
    """Task queue system."""

    def __init__(self, name: str):
        """Initialize queue."""
        self.name = name
        self.redis = redis.Redis.from_url(
            settings.REDIS_URL, encoding="utf-8", decode_responses=True
        )
        self.tasks: Dict[str, Callable] = {}
        self.retention_period = 86400  # 24 hours

    def register_task(self, task_name: str, task_func: Callable):
        """Register task handler."""
        self.tasks[task_name] = task_func

    async def enqueue(self, task_name: str, *args, **kwargs) -> Dict[str, Any]:
        """Add job to queue."""
        try:
            # Create job
            job = Job(
                job_id=str(uuid4()),
                task_name=task_name,
                args=args,
                kwargs=kwargs,
                priority=kwargs.pop("priority", 0),
                retry_count=kwargs.pop("retry_count", 3),
                retry_delay=kwargs.pop("retry_delay", 60),
            )

            # Store job
            await self.redis.set(
                f"job:{job.job_id}",
                json.dumps(job.to_dict()),
                ex=self.retention_period,
            )

            # Add to queue
            await self.redis.zadd(
                f"queue:{self.name}", {job.job_id: job.priority}
            )

            return {"status": "success", "job_id": job.job_id}

        except Exception as e:
            logger.error(f"Enqueue error: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def dequeue(self) -> Optional[Job]:
        """Get next job from queue."""
        try:
            # Get job with highest priority
            result = await self.redis.zpopmax(f"queue:{self.name}")
            if not result:
                return None

            job_id = result[0][0]

            # Get job data
            job_data = await self.redis.get(f"job:{job_id}")
            if not job_data:
                return None

            return Job.from_dict(json.loads(job_data))

        except Exception as e:
            logger.error(f"Dequeue error: {str(e)}")
            return None

    async def process_job(self, job: Job) -> bool:
        """Process job."""
        try:
            # Update job status
            job.status = "processing"
            job.started_at = datetime.now(timezone.utc)
            job.attempts += 1

            await self.redis.set(
                f"job:{job.job_id}",
                json.dumps(job.to_dict()),
                ex=self.retention_period,
            )

            # Get task handler
            task_func = self.tasks.get(job.task_name)
            if not task_func:
                raise ValueError(f"Unknown task: {job.task_name}")

            # Execute task
            result = await task_func(*job.args, **job.kwargs)

            # Update job status
            job.status = "completed"
            job.completed_at = datetime.now(timezone.utc)
            job.result = result

            await self.redis.set(
                f"job:{job.job_id}",
                json.dumps(job.to_dict()),
                ex=self.retention_period,
            )

            return True

        except Exception as e:
            # Handle failure
            job.status = "failed"
            job.error = str(e)

            # Retry if possible
            if job.attempts < job.retry_count:
                await asyncio.sleep(job.retry_delay)
                await self.enqueue(
                    job.task_name,
                    *job.args,
                    priority=job.priority,
                    retry_count=job.retry_count,
                    retry_delay=job.retry_delay,
                    **job.kwargs,
                )

            await self.redis.set(
                f"job:{job.job_id}",
                json.dumps(job.to_dict()),
                ex=self.retention_period,
            )

            logger.error(f"Job processing error: {str(e)}")
            return False

    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job details."""
        try:
            job_data = await self.redis.get(f"job:{job_id}")
            if not job_data:
                return None

            return Job.from_dict(json.loads(job_data))

        except Exception as e:
            logger.error(f"Get job error: {str(e)}")
            return None

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel pending job."""
        try:
            # Get job
            job = await self.get_job(job_id)
            if not job or job.status != "pending":
                return False

            # Remove from queue
            await self.redis.zrem(f"queue:{self.name}", job_id)

            # Update status
            job.status = "cancelled"
            await self.redis.set(
                f"job:{job_id}",
                json.dumps(job.to_dict()),
                ex=self.retention_period,
            )

            return True

        except Exception as e:
            logger.error(f"Cancel job error: {str(e)}")
            return False


class Worker:
    """Background worker."""

    def __init__(self, queue: Queue):
        """Initialize worker."""
        self.queue = queue
        self.running = False

    async def start(self):
        """Start processing jobs."""
        self.running = True

        while self.running:
            try:
                # Get next job
                job = await self.queue.dequeue()
                if not job:
                    await asyncio.sleep(1)
                    continue

                # Process job
                await self.queue.process_job(job)

            except Exception as e:
                logger.error(f"Worker error: {str(e)}")
                await asyncio.sleep(1)

    def stop(self):
        """Stop processing jobs."""
        self.running = False


class QueueService:
    """Background task processing service."""

    def __init__(self):
        """Initialize queue service."""
        self.queues: Dict[str, Queue] = {}
        self.workers: Dict[str, Worker] = {}

    def create_queue(self, name: str) -> Queue:
        """Create new queue."""
        if name not in self.queues:
            self.queues[name] = Queue(name)
        return self.queues[name]

    def get_queue(self, name: str) -> Optional[Queue]:
        """Get existing queue."""
        return self.queues.get(name)

    async def start_worker(self, queue_name: str):
        """Start worker for queue."""
        if queue_name not in self.queues:
            raise ValueError(f"Queue not found: {queue_name}")

        if queue_name not in self.workers:
            worker = Worker(self.queues[queue_name])
            self.workers[queue_name] = worker
            asyncio.create_task(worker.start())

    def stop_worker(self, queue_name: str):
        """Stop worker for queue."""
        if queue_name in self.workers:
            self.workers[queue_name].stop()
            del self.workers[queue_name]

    async def enqueue_task(
        self, queue_name: str, task_name: str, *args, **kwargs
    ) -> Dict[str, Any]:
        """Add task to queue."""
        queue = self.get_queue(queue_name)
        if not queue:
            return {
                "status": "error",
                "message": f"Queue not found: {queue_name}",
            }

        return await queue.enqueue(task_name, *args, **kwargs)

    async def get_job_status(
        self, queue_name: str, job_id: str
    ) -> Dict[str, Any]:
        """Get job status."""
        queue = self.get_queue(queue_name)
        if not queue:
            return {
                "status": "error",
                "message": f"Queue not found: {queue_name}",
            }

        job = await queue.get_job(job_id)
        if not job:
            return {"status": "error", "message": f"Job not found: {job_id}"}

        return {"status": "success", "job": job.to_dict()}

    async def cancel_job(self, queue_name: str, job_id: str) -> Dict[str, Any]:
        """Cancel pending job."""
        queue = self.get_queue(queue_name)
        if not queue:
            return {
                "status": "error",
                "message": f"Queue not found: {queue_name}",
            }

        success = await queue.cancel_job(job_id)
        return {
            "status": "success" if success else "error",
            "message": "Job cancelled" if success else "Failed to cancel job",
        }

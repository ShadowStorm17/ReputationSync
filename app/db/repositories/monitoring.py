"""
Monitoring repository.
Provides database access layer for monitoring model.
"""

from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.repository import Repository
from app.models.database import MonitoringConfig


class MonitoringRepository(Repository[MonitoringConfig]):
    """Monitoring repository."""

    def __init__(self):
        """Initialize repository."""
        super().__init__(MonitoringConfig)

    async def get_by_platform_username(
        self, db: AsyncSession, platform: str, username: str
    ) -> Optional[MonitoringConfig]:
        """Get monitoring config by platform and username."""
        query = select(self.model).filter(
            self.model.platform == platform, self.model.username == username
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def get_by_platform(
        self, db: AsyncSession, platform: str, skip: int = 0, limit: int = 100
    ) -> List[MonitoringConfig]:
        """Get monitoring configs by platform."""
        query = (
            select(self.model)
            .filter(self.model.platform == platform)
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return result.scalars().all()

    async def get_active_configs(
        self, db: AsyncSession, skip: int = 0, limit: int = 100
    ) -> List[MonitoringConfig]:
        """Get active monitoring configs."""
        query = (
            select(self.model)
            .filter(self.model.is_active)
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return result.scalars().all()

    async def get_by_interval(
        self, db: AsyncSession, interval: int, skip: int = 0, limit: int = 100
    ) -> List[MonitoringConfig]:
        """Get monitoring configs by interval."""
        query = (
            select(self.model)
            .filter(self.model.monitoring_interval == interval)
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return result.scalars().all()

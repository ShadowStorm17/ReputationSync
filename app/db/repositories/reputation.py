"""
Reputation repository.
Provides database access layer for reputation model.
"""

from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.repository import Repository
from app.models.database import ReputationScore


class ReputationRepository(Repository[ReputationScore]):
    """Reputation repository."""

    def __init__(self):
        """Initialize repository."""
        super().__init__(ReputationScore)

    async def get_by_platform_username(
        self, db: AsyncSession, platform: str, username: str
    ) -> Optional[ReputationScore]:
        """Get reputation score by platform and username."""
        query = select(self.model).filter(
            self.model.platform == platform, self.model.username == username
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def get_by_platform(
        self, db: AsyncSession, platform: str, skip: int = 0, limit: int = 100
    ) -> List[ReputationScore]:
        """Get reputation scores by platform."""
        query = (
            select(self.model)
            .filter(self.model.platform == platform)
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return result.scalars().all()

    async def get_by_score_range(
        self,
        db: AsyncSession,
        min_score: float,
        max_score: float,
        skip: int = 0,
        limit: int = 100,
    ) -> List[ReputationScore]:
        """Get reputation scores by score range."""
        query = (
            select(self.model)
            .filter(
                self.model.score >= min_score, self.model.score <= max_score
            )
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return result.scalars().all()

    async def get_by_timeframe(
        self, db: AsyncSession, timeframe: str, skip: int = 0, limit: int = 100
    ) -> List[ReputationScore]:
        """Get reputation scores by timeframe."""
        query = (
            select(self.model)
            .filter(self.model.timeframe == timeframe)
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return result.scalars().all()

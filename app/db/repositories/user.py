"""
User repository.
Provides database access layer for user model.
"""

from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.repository import Repository
from app.models.database import User


class UserRepository(Repository[User]):
    """User repository."""

    def __init__(self):
        """Initialize repository."""
        super().__init__(User)

    async def get_by_email(
        self, db: AsyncSession, email: str
    ) -> Optional[User]:
        """Get user by email."""
        return await self.get_by_field(db, "email", email)

    async def get_by_username(
        self, db: AsyncSession, username: str
    ) -> Optional[User]:
        """Get user by username."""
        return await self.get_by_field(db, "username", username)

    async def get_by_api_key(
        self, db: AsyncSession, api_key: str
    ) -> Optional[User]:
        """Get user by API key."""
        return await self.get_by_field(db, "api_key", api_key)

    async def get_active_users(
        self, db: AsyncSession, skip: int = 0, limit: int = 100
    ) -> List[User]:
        """Get active users."""
        query = (
            select(self.model)
            .filter(self.model.is_active)
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return result.scalars().all()

    async def get_users_by_subscription(
        self, db: AsyncSession, tier: str, skip: int = 0, limit: int = 100
    ) -> List[User]:
        """Get users by subscription tier."""
        query = (
            select(self.model)
            .filter(self.model.subscription_tier == tier)
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return result.scalars().all()

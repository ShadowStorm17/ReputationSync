"""
Response repository.
Provides database access layer for response models.
"""

from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.repository import Repository
from app.models.database import AutoResponse, ResponseRule, ResponseTemplate


class ResponseTemplateRepository(Repository[ResponseTemplate]):
    """Response template repository."""

    def __init__(self):
        """Initialize repository."""
        super().__init__(ResponseTemplate)

    async def get_by_category(
        self, db: AsyncSession, category: str, skip: int = 0, limit: int = 100
    ) -> List[ResponseTemplate]:
        """Get templates by category."""
        query = (
            select(self.model)
            .filter(self.model.category == category)
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return result.scalars().all()

    async def get_active_templates(
        self, db: AsyncSession, skip: int = 0, limit: int = 100
    ) -> List[ResponseTemplate]:
        """Get active templates."""
        query = (
            select(self.model)
            .filter(self.model.is_active)
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return result.scalars().all()


class ResponseRuleRepository(Repository[ResponseRule]):
    """Response rule repository."""

    def __init__(self):
        """Initialize repository."""
        super().__init__(ResponseRule)

    async def get_by_trigger_type(
        self,
        db: AsyncSession,
        trigger_type: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[ResponseRule]:
        """Get rules by trigger type."""
        query = (
            select(self.model)
            .filter(self.model.trigger_type == trigger_type)
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return result.scalars().all()

    async def get_by_template(
        self,
        db: AsyncSession,
        template_id: int,
        skip: int = 0,
        limit: int = 100,
    ) -> List[ResponseRule]:
        """Get rules by template."""
        query = (
            select(self.model)
            .filter(self.model.template_id == template_id)
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return result.scalars().all()


class AutoResponseRepository(Repository[AutoResponse]):
    """Auto response repository."""

    def __init__(self):
        """Initialize repository."""
        super().__init__(AutoResponse)

    async def get_by_platform_comment(
        self, db: AsyncSession, platform: str, comment_id: str
    ) -> Optional[AutoResponse]:
        """Get auto response by platform and comment."""
        query = select(self.model).filter(
            self.model.platform == platform,
            self.model.comment_id == comment_id,
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def get_by_platform(
        self, db: AsyncSession, platform: str, skip: int = 0, limit: int = 100
    ) -> List[AutoResponse]:
        """Get auto responses by platform."""
        query = (
            select(self.model)
            .filter(self.model.platform == platform)
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return result.scalars().all()

    async def get_by_status(
        self, db: AsyncSession, status: str, skip: int = 0, limit: int = 100
    ) -> List[AutoResponse]:
        """Get auto responses by status."""
        query = (
            select(self.model)
            .filter(self.model.status == status)
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return result.scalars().all()

    async def get_by_template(
        self,
        db: AsyncSession,
        template_id: int,
        skip: int = 0,
        limit: int = 100,
    ) -> List[AutoResponse]:
        """Get auto responses by template."""
        query = (
            select(self.model)
            .filter(self.model.template_id == template_id)
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return result.scalars().all()


class ResponseRepository:
    ResponseTemplateRepository = ResponseTemplateRepository
    ResponseRuleRepository = ResponseRuleRepository
    AutoResponseRepository = AutoResponseRepository

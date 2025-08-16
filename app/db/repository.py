"""
Database repository.
Provides database access layer.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.database import Base

ModelType = TypeVar("ModelType", bound=Base)


class Repository(Generic[ModelType]):
    """Base repository class."""

    def __init__(self, model: Type[ModelType]):
        """Initialize repository."""
        self.model = model

    async def get(self, db: AsyncSession, id: Any) -> Optional[ModelType]:
        """Get by id."""
        query = select(self.model).filter(self.model.id == id)
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def get_by_field(
        self, db: AsyncSession, field: str, value: Any
    ) -> Optional[ModelType]:
        """Get by field value."""
        query = select(self.model).filter(getattr(self.model, field) == value)
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def get_all(
        self, db: AsyncSession, skip: int = 0, limit: int = 100
    ) -> List[ModelType]:
        """Get all."""
        query = select(self.model).offset(skip).limit(limit)
        result = await db.execute(query)
        return result.scalars().all()

    async def create(
        self, db: AsyncSession, obj_in: Dict[str, Any]
    ) -> ModelType:
        """Create."""
        obj_in["created_at"] = datetime.now(timezone.utc)
        db_obj = self.model(**obj_in)
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj

    async def update(
        self, db: AsyncSession, id: Any, obj_in: Dict[str, Any]
    ) -> Optional[ModelType]:
        """Update."""
        obj_in["updated_at"] = datetime.now(timezone.utc)
        query = (
            update(self.model)
            .where(self.model.id == id)
            .values(**obj_in)
            .returning(self.model)
        )
        result = await db.execute(query)
        await db.commit()
        return result.scalar_one_or_none()

    async def delete(self, db: AsyncSession, id: Any) -> bool:
        """Delete."""
        query = delete(self.model).where(self.model.id == id)
        result = await db.execute(query)
        await db.commit()
        return bool(result.rowcount)

    async def filter(
        self,
        db: AsyncSession,
        filters: Dict[str, Any],
        skip: int = 0,
        limit: int = 100,
    ) -> List[ModelType]:
        """Filter by multiple fields."""
        query = select(self.model)
        for field, value in filters.items():
            query = query.filter(getattr(self.model, field) == value)
        query = query.offset(skip).limit(limit)
        result = await db.execute(query)
        return result.scalars().all()

    async def count(
        self, db: AsyncSession, filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """Count records."""
        query = select(self.model)
        if filters:
            for field, value in filters.items():
                query = query.filter(getattr(self.model, field) == value)
        result = await db.execute(query)
        return len(result.scalars().all())

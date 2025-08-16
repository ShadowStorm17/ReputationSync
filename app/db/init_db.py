"""
Database initialization script.
Creates initial database schema and default data.
"""

import asyncio
from datetime import datetime, timezone
import logging

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings
from app.core.security import get_password_hash
from app.models.database import Base, User

# Get settings
settings = get_settings()

# Create async engine
engine = create_async_engine(
    settings.database.URL,
    echo=settings.database.ECHO,
    pool_size=settings.database.POOL_SIZE,
    max_overflow=settings.database.MAX_OVERFLOW,
    pool_timeout=settings.database.POOL_TIMEOUT,
)

# Create async session factory
async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db():
    """Initialize database."""
    try:
        # Create tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Create default admin user
        async with async_session() as session:
            # Check if admin user exists
            admin = (
                await session.query(User)
                .filter(User.email == "admin@example.com")
                .first()
            )
            if not admin:
                admin = User(
                    email="admin@example.com",
                    username="admin",
                    full_name="Admin User",
                    hashed_password=get_password_hash("admin123"),
                    subscription_tier="admin",
                    is_active=True,
                    created_at=datetime.now(timezone.utc),
                )
                session.add(admin)
                await session.commit()

    except Exception as e:
        logging.getLogger(__name__).error("Error initializing database: %s", str(e))
        raise


if __name__ == "__main__":
    asyncio.run(init_db())

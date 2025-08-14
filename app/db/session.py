"""
Database session management.
Handles database connections and sessions.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.core.config import get_settings

settings = get_settings()

# Create SQLAlchemy engine
engine = create_engine(
    settings.SQLALCHEMY_DATABASE_URI,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create async SQLAlchemy engine
async_engine = create_async_engine(
    settings.SQLALCHEMY_DATABASE_URI.replace('postgresql://', 'postgresql+asyncpg://'),
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

# Create async session factory
async_session = async_sessionmaker(
    bind=async_engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

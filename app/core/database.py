"""
Database module.
Handles database connections, sessions, and operations.
"""

import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Type, TypeVar

from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
from sqlalchemy.orm import Session, scoped_session, sessionmaker
from sqlalchemy.pool import QueuePool

from .config import get_settings
from .constants import DB_MANAGER_NOT_INITIALIZED
from .error_handling import ErrorCategory, ErrorSeverity, ReputationError

logger = logging.getLogger(__name__)
settings = get_settings()

# Type variables for generic operations
T = TypeVar("T")
ModelType = TypeVar("ModelType")


class DatabaseManager:
    """Database manager for handling database operations."""

    def __init__(self):
        """Initialize database manager."""
        self._engine = None
        self._session_factory = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize database connection and session factory."""
        try:
            if self._initialized:
                return

            # Create engine with connection pooling
            self._engine = create_engine(
                settings.database.URL,
                poolclass=QueuePool,
                pool_size=settings.database.POOL_SIZE,
                max_overflow=settings.database.MAX_OVERFLOW,
                pool_timeout=settings.database.POOL_TIMEOUT,
                pool_pre_ping=True,
                echo=settings.database.ECHO,
)

# Create session factory
            self._session_factory = scoped_session(
                sessionmaker(
                    bind=self._engine, autocommit=False, autoflush=False
                )
            )

            # Add event listeners
            self._setup_event_listeners()

            self._initialized = True
            logger.info("Database manager initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise ReputationError(
                message=f"Failed to initialize database: {str(e)}",
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.SYSTEM,
            )

    def _setup_event_listeners(self) -> None:
        """Setup database event listeners."""
        try:

            @event.listens_for(self._engine, "connect")
            def connect(dbapi_connection, connection_record):
                logger.debug("New database connection established")

            @event.listens_for(self._engine, "checkout")
            def checkout(
                dbapi_connection, connection_record, connection_proxy
            ):
                logger.debug("Database connection checked out from pool")

            @event.listens_for(self._engine, "checkin")
            def checkin(dbapi_connection, connection_record):
                logger.debug("Database connection returned to pool")

            @event.listens_for(self._engine, "reset")
            def reset(dbapi_connection, connection_record):
                logger.debug("Database connection reset")

        except Exception as e:
            logger.error(f"Error setting up event listeners: {str(e)}")
            raise ReputationError(
                message=f"Failed to setup database event listeners: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
            )

    @contextmanager
    def get_session(self) -> Session:
        """Get a database session."""
        if not self._initialized:
            raise ReputationError(
                message=DB_MANAGER_NOT_INITIALIZED,
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
            )

        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {str(e)}")
            raise ReputationError(
                message=f"Database operation failed: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
            )
        finally:
            session.close()

    def create_all(self) -> None:
        """Create all database tables."""
        try:
            if not self._initialized:
                raise ReputationError(
                    message=DB_MANAGER_NOT_INITIALIZED,
                    severity=ErrorSeverity.ERROR,
                    category=ErrorCategory.SYSTEM,
                )

            Base.metadata.create_all(self._engine)
            logger.info("Database tables created successfully")

        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
            raise ReputationError(
                message=f"Failed to create database tables: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
            )

    def drop_all(self) -> None:
        """Drop all database tables."""
        try:
            if not self._initialized:
                raise ReputationError(
                    message=DB_MANAGER_NOT_INITIALIZED,
                    severity=ErrorSeverity.ERROR,
                    category=ErrorCategory.SYSTEM,
                )

            Base.metadata.drop_all(self._engine)
            logger.info("Database tables dropped successfully")

        except Exception as e:
            logger.error(f"Error dropping database tables: {str(e)}")
            raise ReputationError(
                message=f"Failed to drop database tables: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
            )

    def get_model(self, model_class: Type[ModelType]) -> ModelType:
        """Get a model instance."""
        return model_class()

    def add(self, model: ModelType) -> None:
        """Add a model instance to the session."""
        try:
            with self.get_session() as session:
                session.add(model)
                session.commit()
                logger.debug(f"Added model instance: {model}")

        except Exception as e:
            logger.error(f"Error adding model instance: {str(e)}")
            raise ReputationError(
                message=f"Failed to add model instance: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
            )

    def update(self, model: ModelType) -> None:
        """Update a model instance."""
        try:
            with self.get_session() as session:
                session.merge(model)
                session.commit()
                logger.debug(f"Updated model instance: {model}")

        except Exception as e:
            logger.error(f"Error updating model instance: {str(e)}")
            raise ReputationError(
                message=f"Failed to update model instance: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
            )

    def delete(self, model: ModelType) -> None:
        """Delete a model instance."""
        try:
            with self.get_session() as session:
                session.delete(model)
                session.commit()
                logger.debug(f"Deleted model instance: {model}")

        except Exception as e:
            logger.error(f"Error deleting model instance: {str(e)}")
            raise ReputationError(
                message=f"Failed to delete model instance: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
            )

    def get_by_id(
        self, model_class: Type[ModelType], id: Any
    ) -> Optional[ModelType]:
        """Get a model instance by ID."""
        try:
            with self.get_session() as session:
                return session.get(model_class, id)

        except Exception as e:
            logger.error(f"Error getting model by ID: {str(e)}")
            raise ReputationError(
                message=f"Failed to get model by ID: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
            )

    def get_all(self, model_class: Type[ModelType]) -> List[ModelType]:
        """Get all instances of a model."""
        try:
            with self.get_session() as session:
                return session.query(model_class).all()

        except Exception as e:
            logger.error(f"Error getting all model instances: {str(e)}")
            raise ReputationError(
                message=f"Failed to get all model instances: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
            )

    def query(self, model_class: Type[ModelType]) -> Any:
        """Get a query object for a model."""
        try:
            with self.get_session() as session:
                return session.query(model_class)

        except Exception as e:
            logger.error(f"Error creating query: {str(e)}")
            raise ReputationError(
                message=f"Failed to create query: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
            )

    def execute(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute a raw SQL query."""
        try:
            with self.get_session() as session:
                return session.execute(query, params or {})

        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise ReputationError(
                message=f"Failed to execute query: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
            )

    def close(self) -> None:
        """Close database connections."""
        try:
            if self._engine:
                self._engine.dispose()
                logger.info("Database connections closed")

        except Exception as e:
            logger.error(f"Error closing database connections: {str(e)}")
            raise ReputationError(
                message=f"Failed to close database connections: {str(e)}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
            )


# Global database manager instance
db_manager = DatabaseManager()

import logging
from app.core.config import get_settings
from app.core.database import engine
from app.models.database import Base

settings = get_settings()
logger = logging.getLogger(__name__)

def init_db():
    """Initialize database and create tables."""
    try:
        # Create all tables
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)

        logger.info("Database initialized successfully!")

    except Exception as e:
        logger.error("Error initializing database: %s", str(e))
        raise


if __name__ == "__main__":
    init_db()

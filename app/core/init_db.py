from app.core.config import get_settings
from app.core.database import engine
from app.models.database import Base

settings = get_settings()


def init_db():
    """Initialize database and create tables."""
    try:
        # Create all tables
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)

        print("Database initialized successfully!")

    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        raise


if __name__ == "__main__":
    init_db()

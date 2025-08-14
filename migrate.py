"""
Database migration script.
Runs database migrations using Alembic.
"""

import os
import sys
import asyncio
from alembic.config import Config
from alembic import command
from app.core.config import get_settings

# Get settings
settings = get_settings()

def run_migrations():
    """Run database migrations."""
    try:
        # Create migrations directory if it doesn't exist
        os.makedirs("app/db/migrations/versions", exist_ok=True)

        # Create Alembic configuration
        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", settings.database.URL)

        # Run migrations
        command.upgrade(alembic_cfg, "head")
        print("Database migrations completed successfully.")

    except Exception as e:
        print(f"Error running database migrations: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_migrations() 
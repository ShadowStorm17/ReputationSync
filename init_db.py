"""
Database initialization script.
Creates initial database schema and default data.
"""

import os
import sys
import asyncio
from app.db.init_db import init_db
from app.core.config import get_settings

# Get settings
settings = get_settings()

def initialize_database():
    """Initialize database."""
    try:
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)

        # Initialize database
        asyncio.run(init_db())
        print("Database initialization completed successfully.")

    except Exception as e:
        print(f"Error initializing database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    initialize_database() 
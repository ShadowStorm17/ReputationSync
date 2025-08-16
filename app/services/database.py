import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import aiosqlite

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DatabaseService:
    def __init__(self):
        self.db_path = settings.DATABASE_URL
        self._pool = None

    async def get_connection(self) -> aiosqlite.Connection:
        """Get a database connection from the pool."""
        if not self._pool:
            self._pool = await aiosqlite.connect(self.db_path)
        return self._pool

    async def close(self):
        """Close the database connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def get_api_key(self, key: str) -> Optional[Dict]:
        """Get API key details."""
        try:
            conn = await self.get_connection()
            async with conn.execute(
                """SELECT id, name, key, user_id, created_at, revoked
                   FROM api_keys WHERE key = ? AND revoked = 0""",
                (key,),
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return {
                        "id": row[0],
                        "name": row[1],
                        "key": row[2],
                        "user_id": row[3],
                        "created_at": row[4],
                        "revoked": row[5],
                    }
                return None
        except Exception as e:
            logger.error("Error getting API key: %s", e)
            return None

    async def create_api_key(
        self, name: str, key: str, user_id: int
    ) -> Optional[Dict]:
        """Create a new API key."""
        try:
            conn = await self.get_connection()
            async with conn.execute(
                """INSERT INTO api_keys (id, name, key, user_id, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    name,
                    key,
                    user_id,
                    datetime.now(timezone.utc).isoformat(),
                ),
            ):
                await conn.commit()
                return await self.get_api_key(key)
        except Exception as e:
            logger.error("Error creating API key: %s", e)
            return None

    async def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        try:
            conn = await self.get_connection()
            async with conn.execute(
                "UPDATE api_keys SET revoked = 1 WHERE id = ?", (key_id,)
            ):
                await conn.commit()
                return True
        except Exception as e:
            logger.error("Error revoking API key: %s", e)
            return False

    async def record_api_usage(
        self, endpoint: str, response_time: int, success: bool, api_key: str
    ) -> bool:
        """Record API usage statistics."""
        try:
            conn = await self.get_connection()
            async with conn.execute(
                """INSERT INTO usage_stats
                   (timestamp, endpoint, response_time, success, api_key)
                   VALUES (datetime('now'), ?, ?, ?, ?)""",
                (endpoint, response_time, 1 if success else 0, api_key),
            ):
                await conn.commit()
                return True
        except Exception as e:
            logger.error("Error recording API usage: %s", e)
            return False

    async def get_usage_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict]:
        """Get API usage statistics."""
        try:
            query = """SELECT endpoint, COUNT(*) as requests,
                      AVG(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100 as success_rate,
                      AVG(response_time) as avg_response_time
                      FROM usage_stats"""
            params = []

            if start_date or end_date:
                query += " WHERE"
                if start_date:
                    query += " timestamp >= ?"
                    params.append(start_date.isoformat())
                if end_date:
                    query += " AND" if start_date else ""
                    query += " timestamp <= ?"
                    params.append(end_date.isoformat())

            query += " GROUP BY endpoint"

            conn = await self.get_connection()
            async with conn.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [
                    {
                        "endpoint": row[0],
                        "requests": row[1],
                        "success_rate": row[2],
                        "avg_response_time": row[3],
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error("Error getting usage stats: %s", e)
            return []


# Global database service instance
db_service = DatabaseService()

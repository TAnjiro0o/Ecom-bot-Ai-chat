"""Database connection management for PostgreSQL."""
from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator
import asyncpg
from fastapi import FastAPI
logger = logging.getLogger(__name__)
CREATE_PRODUCTS_TABLE_SQL = """
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS products (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    category TEXT NOT NULL,
    price DOUBLE PRECISION NOT NULL CHECK (price >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""
class DatabaseManager:
    """Thin wrapper around an asyncpg connection pool."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Create the connection pool and initialize schema."""
        self.pool = await asyncpg.create_pool(dsn=self._dsn, min_size=1, max_size=10)
        async with self.pool.acquire() as connection:
            await connection.execute(CREATE_PRODUCTS_TABLE_SQL)
        logger.info("PostgreSQL pool initialized")

    async def disconnect(self) -> None:
        """Close the connection pool."""
        if self.pool is not None:
            await self.pool.close()
            self.pool = None
            logger.info("PostgreSQL pool closed")

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[asyncpg.Connection]:
        """Yield a single database connection from the pool."""
        if self.pool is None:
            raise RuntimeError("Database pool is not initialized")

        async with self.pool.acquire() as connection:
            yield connection


def get_db(app: FastAPI) -> DatabaseManager:
    """Return the database manager stored on FastAPI app state."""
    return app.state.db

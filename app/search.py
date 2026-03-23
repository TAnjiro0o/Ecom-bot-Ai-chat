"""Search orchestration logic."""

from __future__ import annotations

import logging
import time

from app.crud import get_products_by_ids
from app.db import DatabaseManager
from app.elastic import ElasticsearchManager
from app.embedding import get_embedding
from app.schemas import ProductRead

logger = logging.getLogger(__name__)

SEARCH_CACHE_SIZE = 128
_search_cache: dict[str, list[str]] = {}


class SearchBackendUnavailableError(Exception):
    """Raised when Elasticsearch or embedding generation is unavailable."""


class DatabaseUnavailableError(Exception):
    """Raised when PostgreSQL hydration fails."""


def _cache_get(query: str) -> list[str] | None:
    return _search_cache.get(query)


def _cache_set(query: str, ids: list[str]) -> None:
    if len(_search_cache) >= SEARCH_CACHE_SIZE:
        oldest_key = next(iter(_search_cache))
        _search_cache.pop(oldest_key, None)
    _search_cache[query] = ids


async def search_products(
    *,
    query: str,
    db: DatabaseManager,
    elastic: ElasticsearchManager,
) -> list[ProductRead]:
    """Run hybrid search and hydrate ranked PostgreSQL records."""
    started = time.perf_counter()
    cached_ids = _cache_get(query)

    if cached_ids is None:
        try:
            query_embedding = get_embedding(query)
            hits = await elastic.search_products(query=query, embedding=query_embedding, size=50)
            ranked_ids = [hit["_id"] for hit in hits][:20]
            _cache_set(query, ranked_ids)
        except Exception as exc:
            raise SearchBackendUnavailableError from exc
    else:
        ranked_ids = cached_ids[:20]

    if not ranked_ids:
        logger.info("Search query='%s' returned no hits", query)
        return []

    try:
        async with db.connection() as connection:
            records = await get_products_by_ids(connection, ranked_ids)
    except Exception as exc:
        raise DatabaseUnavailableError from exc

    records_by_id = {str(record.id): record for record in records}
    ordered = [records_by_id[product_id] for product_id in ranked_ids if product_id in records_by_id]

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info("Search query='%s' returned %s rows in %.2fms", query, len(ordered), elapsed_ms)
    return [ProductRead.model_validate(record) for record in ordered]


def clear_search_cache() -> None:
    """Invalidate the in-memory search cache."""
    _search_cache.clear()

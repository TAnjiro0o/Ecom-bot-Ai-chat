"""Elasticsearch integration for indexing and search."""

from __future__ import annotations

import logging
from typing import Any

from elasticsearch import AsyncElasticsearch

from app.embedding import EMBEDDING_DIMENSIONS

logger = logging.getLogger(__name__)

INDEX_NAME = "products"


PRODUCTS_INDEX_MAPPING: dict[str, Any] = {
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        }
    },
    "mappings": {
        "properties": {
            "name": {"type": "text"},
            "description": {"type": "text"},
            "category": {"type": "keyword"},
            "price": {"type": "float"},
            "stock":{"type": "integer"},
            "embedding": {
                "type": "dense_vector",
                "dims": EMBEDDING_DIMENSIONS,
                "index": True,
                "similarity": "cosine",
            },
        }
    },
}


class ElasticsearchManager:
    """Wrapper around AsyncElasticsearch."""

    def __init__(self, url: str) -> None:
        self.client = AsyncElasticsearch(url,headers={"Accept": "application/json"},)

    async def ensure_index(self) -> None:
        try:
         exists = await self.client.indices.exists(index=INDEX_NAME)
        except Exception:
         exists = False

        if not exists:
         await self.client.indices.create(index=INDEX_NAME, body=PRODUCTS_INDEX_MAPPING)
         logger.info("Created Elasticsearch index '%s'", INDEX_NAME)

    async def close(self) -> None:
        """Close the Elasticsearch transport."""
        await self.client.close()

    async def index_product(self, product_id: str, document: dict[str, Any]) -> None:
        """Index a product document."""
        await self.client.index(index=INDEX_NAME, id=product_id, document=document, refresh="wait_for")

    async def search_products(self, query: str, embedding: list[float], size: int = 50) -> list[dict[str, Any]]:
      body = {
        "size": size,
        "_source": False,
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["name^3", "description^2", "category"],
                "fuzziness": "AUTO",
            }
        },
        "knn": {
            "field": "embedding",
            "query_vector": embedding,
            "k": size,
            "num_candidates": max(size, 50),
        },
    }

      response = await self.client.search(index=INDEX_NAME, body=body)
      return response["hits"]["hits"]
"""FastAPI entrypoint for the product search engine."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from app.schemas import CartUpdateRequest

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware

from app import crud
from app.seed_data import main as seed_main
from app.db import DatabaseManager, get_db
from app.elastic import ElasticsearchManager
from app.embedding import build_product_text, get_embedding
from app.schemas import (
    ProductCreate,
    ProductCreateResponse,
    ProductRead,
    SearchResponse,
    CartAddRequest,
    CartProductItem,
    CartResponse,
    WishlistProductItem,
    WishlistAddRequest,
    WishlistResponse,
)
from app.search import (
    DatabaseUnavailableError,
    SearchBackendUnavailableError,
    clear_search_cache,
    search_products,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:0707@localhost:5432/product_search",
    )
    elasticsearch_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")

    app.state.db = DatabaseManager(database_url)
    app.state.elastic = ElasticsearchManager(elasticsearch_url)

    await app.state.db.connect()
    await app.state.elastic.ensure_index()

    try:
        yield
    finally:
        await app.state.elastic.close()
        await app.state.db.disconnect()


app = FastAPI(
    title="Product Search Engine",
    version="1.0.0",
    lifespan=lifespan,
)

@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "%s %s completed with %s in %.2fms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/seed")
async def seed():
    await seed_main()
    return {"message": "Database seeded successfully"}

@app.post("/products", response_model=ProductCreateResponse, status_code=201)
async def create_product(payload: ProductCreate, request: Request):
    db = get_db(request.app)
    elastic: ElasticsearchManager = request.app.state.elastic

    async with db.connection() as connection:
        try:
            product = await crud.create_product(connection, payload)
        except Exception as exc:
            logger.exception("DB error: %s", exc)
            raise HTTPException(503, "Database unavailable")

    indexed = True
    message = "Product created and indexed successfully."

    try:
        text = build_product_text(product.name, product.description)
        embedding = get_embedding(text)

        await elastic.index_product(
            product_id=str(product.id),
            document={
                "name": product.name,
                "description": product.description,
                "category": product.category,
                "price": product.price,
                "embedding": embedding,
            },
        )

        clear_search_cache()

    except Exception as exc:
        indexed = False
        message = "Created but indexing failed"
        logger.exception("Elastic error: %s", exc)

    return ProductCreateResponse(
        product=ProductRead.model_validate(product),
        indexed_in_search=indexed,
        message=message,
    )

@app.get("/search", response_model=SearchResponse)
async def search(request: Request, q: str = Query(..., min_length=1)):
    db = get_db(request.app)
    elastic: ElasticsearchManager = request.app.state.elastic

    try:
       results = await search_products(
    query=q,
    db=db,
    elastic=elastic,
)

    except Exception as exc:
        print(" REAL ERROR:", exc)
        raise HTTPException(500, f"Search failed: {str(exc)}")

    return SearchResponse(query=q, count=len(results), results=results)

@app.post("/cart/add", response_model=CartProductItem, status_code=201)
async def add_to_cart(payload: CartAddRequest, request: Request):
    db = get_db(request.app)

    async with db.connection() as connection:
        product = await crud.get_product_by_id(connection, payload.product_id)
        if not product:
            raise HTTPException(404, "Product not found")
        if product.stock <= 0:
         raise HTTPException(400, "Out of stock")
        item = await crud.add_to_cart(connection, payload)
    return CartProductItem(
        id=item.id,
        user_id=item.user_id,
        product_id=item.product_id,
        quantity=item.quantity,
        stock=item.product.stock,
        created_at=item.created_at,
        product=ProductRead.model_validate(item.product),
    )

@app.post("/cart/remove", status_code=200)
async def remove_from_cart(payload: CartAddRequest, request: Request):
    db = get_db(request.app)

    async with db.connection() as connection:
        try:
            await crud.remove_from_cart(
                connection, payload.user_id, payload.product_id
            )
        except Exception as exc:
            logger.exception("Remove failed: %s", exc)
            raise HTTPException(503, "Failed to remove item")

    return {"message": "Item removed"}


@app.get("/cart", response_model=CartResponse)
async def get_cart(user_id: str, request: Request):
    db = get_db(request.app)

    async with db.connection() as connection:
        items = await crud.get_cart(connection, user_id)

    return CartResponse(
        user_id=user_id,
        items=[
            CartProductItem(
                id=i.id,
                user_id=i.user_id,
                product_id=i.product_id,
                quantity=i.quantity,
                stock=i.product.stock,
                created_at=i.created_at,
                product=ProductRead.model_validate(i.product),
            )
            for i in items
        ],
    )
@app.post("/cart/update")
async def update_cart(payload: CartUpdateRequest, request: Request):
    db = get_db(request.app)

    async with db.connection() as conn:
        product = await crud.get_product_by_id(conn, payload.product_id)

        try:
            item = await crud.get_cart_item(
                conn, payload.user_id, payload.product_id
            )

            if not item:
                raise HTTPException(404, "Item not found")

            new_qty = item["quantity"] + payload.quantity

            if new_qty <= 0:
                await crud.remove_from_cart(
                    conn, payload.user_id, payload.product_id
                )
                return {"status": "removed"}
            if new_qty > product.stock:
                raise HTTPException(400, "Stock limit exceeded")

            updated = await crud.update_cart_quantity(
                conn,
                payload.user_id,
                payload.product_id,
                new_qty,
            )

            return {"status": "updated", "quantity": new_qty}

        except Exception as exc:
             print(" REAL ERROR:", exc)
             raise HTTPException(500, str(exc)) from exc

@app.post("/wishlist/add", response_model=WishlistProductItem, status_code=201)
async def add_to_wishlist(payload: WishlistAddRequest, request: Request):
    db = get_db(request.app)

    async with db.connection() as connection:
        product = await crud.get_product_by_id(connection, payload.product_id)
        if not product:
            raise HTTPException(404, "Product not found")

        item = await crud.add_to_wishlist(connection, payload)

    return WishlistProductItem(
        id=item.id,
        user_id=item.user_id,
        product_id=item.product_id,
        created_at=item.created_at,
        product=ProductRead.model_validate(item.product),
    )


@app.get("/wishlist", response_model=WishlistResponse)
async def get_wishlist(user_id: str, request: Request):
    db = get_db(request.app)

    async with db.connection() as connection:
        items = await crud.get_wishlist(connection, user_id)

    return WishlistResponse(
        user_id=user_id,
        items=[
            WishlistProductItem(
                id=i.id,
                user_id=i.user_id,
                product_id=i.product_id,
                created_at=i.created_at,
                product=ProductRead.model_validate(i.product),
            )
            for i in items
        ],
    )
@app.post("/wishlist/remove")
async def remove_from_wishlist(payload: WishlistAddRequest, request: Request):
    db = get_db(request.app)

    async with db.connection() as conn:
        await crud.remove_from_wishlist(
            conn,
            payload.user_id,
            payload.product_id
        )

    return {"status": "removed"}
@app.post("/cart/checkout")
async def checkout(request: Request, user_id: str):
    db = get_db(request.app)

    async with db.connection() as conn:
        items = await crud.get_cart(conn, user_id)

        if not items:
            raise HTTPException(400, "Cart is empty")

        for item in items:
            if item.quantity > item.product.stock:
                raise HTTPException(
                    400,
                    f"{item.product.name} out of stock"
                )

        
        for item in items:
            await conn.execute(
                """
                UPDATE products
                SET stock = stock - $1
                WHERE id = $2::uuid
                """,
                item.quantity,
                str(item.product_id),
            )

        
        await conn.execute(
            """
            DELETE FROM cart WHERE user_id = $1
            """,
            user_id,
        )

    return {"status": "order placed"}
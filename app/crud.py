"""CRUD operations for products."""
from __future__ import annotations
from typing import Iterable
from uuid import UUID
import asyncpg
from app.models import ProductRecord, CartItemRecord, WishlistItemRecord
from app.schemas import ProductCreate, WishlistAddRequest, CartAddRequest

def _row_to_product(row: asyncpg.Record) -> ProductRecord:
    return ProductRecord(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        category=row["category"],
        price=row["price"],
        stock=row["stock"],
        created_at=row["created_at"],
    )

async def create_product(connection: asyncpg.Connection, payload: ProductCreate) -> ProductRecord:
    row = await connection.fetchrow(
        """
        INSERT INTO products (name, description, category, price, stock)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id, name, description, category, price, stock, created_at
        """,
        payload.name,
        payload.description,
        payload.category,
        payload.price,
        payload.stock,
    )
    if row is None:
        raise RuntimeError("Failed to insert product")

    return _row_to_product(row)

async def get_products_by_ids(connection: asyncpg.Connection, ids: Iterable[str]) -> list[ProductRecord]:
    ids_list = [UUID(product_id) for product_id in ids]
    if not ids_list:
        return []

    rows = await connection.fetch(
        """
        SELECT id, name, description, category, price, stock, created_at
        FROM products
        WHERE id = ANY($1::uuid[])
        """,
        ids_list,
    )
    return [_row_to_product(row) for row in rows]

async def get_product_by_id(connection: asyncpg.Connection, product_id: str | UUID) -> ProductRecord | None:
    row = await connection.fetchrow(
        """
        SELECT id, name, description, category, price, stock, created_at
        FROM products
        WHERE id = $1::uuid
        """,
        str(product_id),
    )
    return _row_to_product(row) if row else None

async def add_to_cart(connection: asyncpg.Connection, payload: CartAddRequest) -> CartItemRecord:
    product = await get_product_by_id(connection, payload.product_id)

    if product is None:
        raise RuntimeError("Product not found")
    existing = await connection.fetchrow(
        """
        SELECT quantity FROM cart
        WHERE user_id = $1 AND product_id = $2::uuid
        """,
        payload.user_id,
        str(payload.product_id),
    )

    existing_qty = existing["quantity"] if existing else 0
    new_qty = existing_qty + payload.quantity

    if new_qty > product.stock:
        raise RuntimeError("Stock limit exceeded")

    row = await connection.fetchrow(
        """
        INSERT INTO cart (user_id, product_id, quantity)
        VALUES ($1, $2::uuid, $3)
        ON CONFLICT (user_id, product_id)
        DO UPDATE SET quantity = cart.quantity + EXCLUDED.quantity
        RETURNING id, user_id, product_id, quantity, created_at
        """,
        payload.user_id,
        str(payload.product_id),
        payload.quantity,
    )

    return CartItemRecord(
        id=row["id"],
        user_id=row["user_id"],
        product_id=row["product_id"],
        quantity=row["quantity"],
        created_at=row["created_at"],
        product=product,
    )


async def get_cart(connection: asyncpg.Connection, user_id: str) -> list[CartItemRecord]:
    rows = await connection.fetch(
        """
        SELECT
            c.id,
            c.user_id,
            c.product_id,
            c.quantity,
            c.created_at,

            p.id AS product_id_full,
            p.name AS product_name,
            p.description AS product_description,
            p.category AS product_category,
            p.price AS product_price,
            p.stock AS product_stock,
            p.created_at AS product_created_at

        FROM cart c
        INNER JOIN products p ON p.id = c.product_id
        WHERE c.user_id = $1
        ORDER BY c.created_at DESC
        """,
        user_id,
    )

    items: list[CartItemRecord] = []
    for row in rows:
        items.append(
            CartItemRecord(
                id=row["id"],
                user_id=row["user_id"],
                product_id=row["product_id"],
                quantity=row["quantity"],
                created_at=row["created_at"],
                product=ProductRecord(
                    id=row["product_id_full"],
                    name=row["product_name"],
                    description=row["product_description"],
                    category=row["product_category"],
                    price=row["product_price"],
                    stock=row["product_stock"], 
                    created_at=row["product_created_at"],
                ),
            )
        )
    return items


async def remove_from_cart(connection, user_id: str, product_id: str):
    await connection.execute(
        """
        DELETE FROM cart
        WHERE user_id = $1 AND product_id = $2::uuid
        """,
        user_id,
        product_id,
    )


async def get_cart_item(conn, user_id: str, product_id: str):
    return await conn.fetchrow(
        """
        SELECT * FROM cart
        WHERE user_id = $1 AND product_id = $2::uuid
        """,
        user_id,
        product_id,
    )


async def update_cart_quantity(conn, user_id: str, product_id: str, quantity: int):
    product = await get_product_by_id(conn, product_id)

    if quantity > product.stock:
        raise RuntimeError("Stock limit exceeded")

    result = await conn.fetchrow(
        """
        UPDATE cart
        SET quantity = $1
        WHERE user_id = $2 AND product_id = $3::uuid
        RETURNING *
        """,
        quantity,
        user_id,
        product_id,
    )

    if result is None:
        raise Exception("Update failed")

    return result

async def add_to_wishlist(connection: asyncpg.Connection, payload: WishlistAddRequest) -> WishlistItemRecord:
    row = await connection.fetchrow(
        """
        INSERT INTO wishlist (user_id, product_id)
        VALUES ($1, $2::uuid)
        ON CONFLICT (user_id, product_id)
        DO UPDATE SET user_id = EXCLUDED.user_id
        RETURNING id, user_id, product_id, created_at
        """,
        payload.user_id,
        str(payload.product_id),
    )

    if row is None:
        raise RuntimeError("Failed to add to wishlist")

    product = await get_product_by_id(connection, row["product_id"])

    return WishlistItemRecord(
        id=row["id"],
        user_id=row["user_id"],
        product_id=row["product_id"],
        created_at=row["created_at"],
        product=product,
    )

async def get_wishlist(connection: asyncpg.Connection, user_id: str) -> list[WishlistItemRecord]:
    rows = await connection.fetch(
        """
        SELECT
            w.id,
            w.user_id,
            w.product_id,
            w.created_at,

            p.id AS product_id_full,
            p.name AS product_name,
            p.description AS product_description,
            p.category AS product_category,
            p.price AS product_price,
            p.stock AS product_stock,
            p.created_at AS product_created_at

        FROM wishlist w
        INNER JOIN products p ON p.id = w.product_id
        WHERE w.user_id = $1
        ORDER BY w.created_at DESC
        """,
        user_id,
    )

    items: list[WishlistItemRecord] = []
    for row in rows:
        items.append(
            WishlistItemRecord(
                id=row["id"],
                user_id=row["user_id"],
                product_id=row["product_id"],
                created_at=row["created_at"],
                product=ProductRecord(
                    id=row["product_id_full"],
                    name=row["product_name"],
                    description=row["product_description"],
                    category=row["product_category"],
                    price=row["product_price"],
                    stock=row["product_stock"],
                    created_at=row["product_created_at"],
                ),
            )
        )
    return items

async def remove_from_wishlist(conn, user_id: str, product_id: str):
    await conn.execute(
        """
        DELETE FROM wishlist
        WHERE user_id = $1 AND product_id = $2::uuid
        """,
        user_id,
        product_id,
    )
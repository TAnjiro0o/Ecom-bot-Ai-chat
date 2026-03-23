"""Domain model constants and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

@dataclass(slots=True)
class ProductRecord:
    """Internal product representation fetched from PostgreSQL."""

    id: UUID
    name: str
    description: str
    category: str
    price: float
    stock: int  
    created_at: datetime

@dataclass(slots=True)
class CartItemRecord:
    """Cart item row joined with product data."""

    id: UUID
    user_id: str
    product_id: UUID
    quantity: int
    created_at: datetime
    product: ProductRecord  



@dataclass(slots=True)
class WishlistItemRecord:
    """Wishlist row joined with product data."""

    id: UUID
    user_id: str
    product_id: UUID
    created_at: datetime
    product: ProductRecord  
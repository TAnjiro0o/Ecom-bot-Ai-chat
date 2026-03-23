"""Pydantic schemas for request and response models."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from pydantic import BaseModel

class CartUpdateRequest(BaseModel):
    user_id: str
    product_id: str
    quantity: int  


class ProductCreate(BaseModel):
    """Payload for creating a product."""

    name: str = Field(min_length=1, max_length=255)
    description: str = Field(min_length=1, max_length=5000)
    category: str = Field(min_length=1, max_length=255)
    price: float = Field(ge=0)
    stock: int


class ProductRead(BaseModel):
    """API representation of a product."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str
    category: str
    price: float
    created_at: datetime
    stock: int


class ProductCreateResponse(BaseModel):
    """Response returned after attempting to create a product."""

    product: ProductRead
    indexed_in_search: bool
    message: str


class SearchResponse(BaseModel):
    """Response returned by the search endpoint."""

    query: str
    count: int
    results: list[ProductRead]

class CartAddRequest(BaseModel):
    """Add-to-cart payload."""

    user_id: str = Field(min_length=1, max_length=128)
    product_id: UUID
    quantity: int = Field(default=1, ge=1, le=20)


class WishlistAddRequest(BaseModel):
    """Add-to-wishlist payload."""

    user_id: str = Field(min_length=1, max_length=128)
    product_id: UUID


class CartProductItem(BaseModel):
    """Cart item including product details."""

    id: UUID
    user_id: str
    product_id: UUID
    quantity: int
    created_at: datetime
    product: ProductRead


class WishlistProductItem(BaseModel):
    """Wishlist item including product details."""

    id: UUID
    user_id: str
    product_id: UUID
    created_at: datetime
    product: ProductRead


class CartResponse(BaseModel):
    """Cart response."""

    user_id: str
    items: list[CartProductItem]


class WishlistResponse(BaseModel):
    """Wishlist response."""

    user_id: str
    items: list[WishlistProductItem]


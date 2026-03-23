"""Seed realistic branded product data into PostgreSQL + Elasticsearch with random stock"""

import asyncio
import random

from app.db import DatabaseManager
from app import crud
from app.elastic import ElasticsearchManager
from app.embedding import build_product_text, get_embedding
from app.schemas import ProductCreate

DATABASE_URL = "postgresql://postgres:0707@localhost:5432/product_search"
ELASTIC_URL = "http://localhost:9200"

PRODUCTS = [
    # FOOTWEAR
    {"name": "Nike Air Zoom Pegasus 40", "description": "Premium running shoes", "category": "footwear", "price": 9999},
    {"name": "Adidas Ultraboost 22", "description": "Energy return running shoes", "category": "footwear", "price": 11999},
    {"name": "Puma RS-X Sneakers", "description": "Stylish chunky sneakers", "category": "footwear", "price": 6999},
    {"name": "Reebok Nano X3", "description": "Training shoes for gym", "category": "footwear", "price": 8499},
    {"name": "Skechers Go Walk", "description": "Comfortable walking shoes", "category": "footwear", "price": 4999},

    # ELECTRONICS
    {"name": "Apple iPhone 15", "description": "Latest iPhone with A16 chip", "category": "electronics", "price": 79900},
    {"name": "Samsung Galaxy S23", "description": "Flagship Android smartphone", "category": "electronics", "price": 74999},
    {"name": "OnePlus 12", "description": "Fast AMOLED smartphone", "category": "electronics", "price": 64999},
    {"name": "MacBook Air M2", "description": "Lightweight Apple laptop", "category": "electronics", "price": 109900},
    {"name": "Dell XPS 13", "description": "Premium ultrabook laptop", "category": "electronics", "price": 99999},

    {"name": "Sony WH-1000XM5", "description": "Noise cancelling headphones", "category": "electronics", "price": 29999},
    {"name": "JBL Flip 6", "description": "Portable Bluetooth speaker", "category": "electronics", "price": 9999},
    {"name": "Apple AirPods Pro", "description": "Wireless earbuds with ANC", "category": "electronics", "price": 24900},
    {"name": "Samsung 55\" QLED TV", "description": "4K Smart TV", "category": "electronics", "price": 79999},
    {"name": "Amazon Echo Dot", "description": "Smart speaker with Alexa", "category": "electronics", "price": 4499},

    # CLOTHING
    {"name": "Levi's 511 Slim Jeans", "description": "Classic slim fit jeans", "category": "clothing", "price": 3999},
    {"name": "Nike Dri-FIT T-Shirt", "description": "Breathable sports t-shirt", "category": "clothing", "price": 1999},
    {"name": "Adidas Track Jacket", "description": "Athletic jacket", "category": "clothing", "price": 4999},
    {"name": "Zara Casual Shirt", "description": "Slim fit casual shirt", "category": "clothing", "price": 2999},
    {"name": "H&M Hoodie", "description": "Soft cotton hoodie", "category": "clothing", "price": 2499},

    {"name": "Uniqlo Polo Shirt", "description": "Minimal premium polo", "category": "clothing", "price": 1999},
    {"name": "Allen Solly Formal Shirt", "description": "Office wear shirt", "category": "clothing", "price": 2799},
    {"name": "Peter England Trousers", "description": "Formal trousers", "category": "clothing", "price": 2499},
    {"name": "Decathlon Gym Shorts", "description": "Workout shorts", "category": "clothing", "price": 999},
    {"name": "Nike Running Shorts", "description": "Running performance shorts", "category": "clothing", "price": 1499},

    # HOME
    {"name": "Philips Air Fryer", "description": "Healthy oil-free cooking", "category": "home", "price": 8999},
    {"name": "Dyson V11 Vacuum Cleaner", "description": "High suction vacuum", "category": "home", "price": 45999},
    {"name": "LG 7kg Washing Machine", "description": "Front load washer", "category": "home", "price": 32999},
    {"name": "Samsung Refrigerator 253L", "description": "Double door fridge", "category": "home", "price": 27999},
    {"name": "Prestige Mixer Grinder", "description": "Kitchen appliance", "category": "home", "price": 3499},

    {"name": "Havells Ceiling Fan", "description": "Energy efficient fan", "category": "home", "price": 2499},
    {"name": "Syska LED Bulb Pack", "description": "Pack of 4 LED bulbs", "category": "home", "price": 399},
    {"name": "Godrej Microwave Oven", "description": "Convection microwave", "category": "home", "price": 7999},
    {"name": "Milton Water Bottle", "description": "Steel insulated bottle", "category": "home", "price": 899},
    {"name": "Sleepyhead Mattress", "description": "Comfort foam mattress", "category": "home", "price": 12999},

    # BOOKS
    {"name": "Atomic Habits", "description": "Build better habits", "category": "books", "price": 499},
    {"name": "Rich Dad Poor Dad", "description": "Finance classic", "category": "books", "price": 399},
    {"name": "The Alchemist", "description": "Famous fiction novel", "category": "books", "price": 299},
    {"name": "Deep Learning Book", "description": "AI and ML concepts", "category": "books", "price": 899},
    {"name": "Clean Code", "description": "Software engineering practices", "category": "books", "price": 799},

    {"name": "The Psychology of Money", "description": "Money behavior insights", "category": "books", "price": 499},
    {"name": "Zero to One", "description": "Startup innovation", "category": "books", "price": 450},
    {"name": "Think and Grow Rich", "description": "Success mindset", "category": "books", "price": 350},
    {"name": "Sapiens", "description": "Human history", "category": "books", "price": 699},
    {"name": "Harry Potter and the Sorcerer's Stone", "description": "Fantasy novel", "category": "books", "price": 399},
]

async def main():
    db = DatabaseManager(DATABASE_URL)
    elastic = ElasticsearchManager(ELASTIC_URL)

    print(" Starting seed with random stock...")

    await db.connect()
    await elastic.ensure_index()

    async with db.connection() as conn:
        for p in PRODUCTS:
            stock = random.randint(0, 25)  

            product_data = ProductCreate(**p, stock=stock)

            product = await crud.create_product(conn, product_data)

            text = build_product_text(product.name, product.description)
            embedding = get_embedding(text)

            await elastic.index_product(
                product_id=str(product.id),
                document={
                    "name": product.name,
                    "description": product.description,
                    "category": product.category,
                    "price": product.price,
                    "stock": product.stock,
                    "embedding": embedding,
                },
            )

            print(f" Inserted: {product.name} (stock={stock})")

    await elastic.close()
    await db.disconnect()

    print(" Seeding completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
from typing import List, Dict, Any, Literal, Optional
from dataclasses import dataclass

DatabaseType = Literal["products", "support", "finance"]


@dataclass
class CollectionConfig:
    name: str
    description: str
    collection_name: str  # This will be used as Qdrant collection name


# Collection configurations
COLLECTIONS: Dict[DatabaseType, CollectionConfig] = {
    "products": CollectionConfig(
        name="Product Information",
        description="Product details, specifications, and features",
        collection_name="products_collection"
    ),
    "support": CollectionConfig(
        name="Customer Support & FAQ",
        description="Customer support information, frequently asked questions, and guides",
        collection_name="support_collection"
    ),
    "finance": CollectionConfig(
        name="Financial Information",
        description="Financial data, revenue, costs, and liabilities",
        collection_name="finance_collection"
    )
}
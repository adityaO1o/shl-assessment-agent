"""Quick test script for the catalog loader.

This script loads the catalog using `load_catalog` and prints a
small summary. It also asserts the catalog is not empty.
"""
from typing import List
import sys
from pathlib import Path

# Ensure workspace root is on sys.path so `app` package can be imported
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.catalog_loader import CatalogItem, load_catalog


def main() -> None:
    """Load the catalog, print summary fields, and assert non-empty."""
    items: List[CatalogItem] = load_catalog()
    total = len(items)
    print(f"Total assessments: {total}")

    if total:
        first = items[0]
        print(f"First assessment name: {first.name}")
        print(f"First assessment searchable_text: {first.searchable_text}")

    # Verification
    assert total > 0, "Catalog is empty"
    print("Success: catalog loaded and contains entries")


if __name__ == "__main__":
    main()

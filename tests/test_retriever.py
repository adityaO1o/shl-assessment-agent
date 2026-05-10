"""Smoke test for `SHLRetriever`.

Builds the FAISS index and runs a sample semantic search, printing the
top results with similarity scores.
"""
from pathlib import Path
import sys
from typing import List, Tuple

# Ensure workspace root is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retriever import SHLRetriever


def main() -> None:
    retriever = SHLRetriever()
    print("Building FAISS index (this may download models and take a moment)...")
    retriever.build_index()

    query = "senior java backend engineer spring aws docker"
    print(f"Searching for: {query}\n")
    results: List[Tuple] = retriever.search(query, top_k=5)

    if not results:
        print("No results returned. Check that faiss and sentence-transformers are installed.")
        return

    print("Top results:")
    for rank, (item, score) in enumerate(results, start=1):
        print(f"{rank}. {item.name} — similarity={score:.4f}")

    # Basic verification
    assert len(results) > 0, "Expected at least one search result"
    print("Success: retriever returned results")


if __name__ == "__main__":
    main()

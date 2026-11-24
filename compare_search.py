# compare_search.py  -> usage: python compare_search.py "your question"
import sys

from app.retrieve import bm25_search, hybrid_search, vector_search

query = sys.argv[1]


def show(title, results):
    print(f"\n=== {title} ===")
    for rank, c in enumerate(results[:5], start=1):
        print(f"{rank}. {c['meta']['doc_id']:<30} | {c['text'][:60]!r}")


show("VECTOR (meaning)", vector_search(query, 20))
show("BM25 (exact words)", bm25_search(query, 20))
show("HYBRID (merged)", hybrid_search(query, 20))
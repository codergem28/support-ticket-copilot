# app/retrieve.py
from app.store import col, embed_query


def vector_search(query: str, k: int = 5):
    r = col.query(query_embeddings=[embed_query(query)], n_results=k)
    return [
        {"id": i, "text": d, "meta": m, "score": 1 - dist}
        for i, d, m, dist in zip(
            r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]
        )
    ]
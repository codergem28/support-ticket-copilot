# app/cache.py
import hashlib
import json

from app.store import client, embed_query

cache = client.get_or_create_collection("cache", metadata={"hnsw:space": "cosine"})

THRESHOLD = 0.92   # how similar two questions must be to reuse an answer (we will tune this)


def cache_lookup(question: str):
    """Return (similarity, saved answer) of the closest stored question, or (None, None)."""
    if cache.count() == 0:
        return None, None
    r = cache.query(query_embeddings=[embed_query(question)], n_results=1)
    similarity = 1 - r["distances"][0][0]
    payload = json.loads(r["metadatas"][0][0]["payload"])
    return similarity, payload


def cache_get(question: str):
    similarity, payload = cache_lookup(question)
    if similarity is not None and similarity >= THRESHOLD:
        return {**payload, "cache_sim": similarity}
    return None


def cache_put(question: str, payload: dict):
    cache.upsert(
        ids=[hashlib.sha1(question.encode()).hexdigest()],
        documents=[question],
        embeddings=[embed_query(question)],
        metadatas=[{"payload": json.dumps(payload)}],
    )
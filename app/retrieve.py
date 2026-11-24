import re

from rank_bm25 import BM25Okapi
from app.store import col, embed_query


def vector_search(query: str, k: int = 5):
    r = col.query(query_embeddings=[embed_query(query)], n_results=k)
    return [
        {"id": i, "text": d, "meta": m, "score": 1 - dist}
        for i, d, m, dist in zip(
            r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]
        )
    ]

# ---------- 2. keyword search (BM25) ----------
_bm25 = None          # the keyword index, built once and kept in memory
_ids, _docs, _metas = [], [], []


def tokenize(text: str):
    # lowercase the text and cut it into words: "Hello, World!" -> ["hello", "world"]
    return re.findall(r"\w+", text.lower())


def build_bm25():
    # read every stored chunk from the database and build the keyword index
    global _bm25, _ids, _docs, _metas
    data = col.get()
    _ids, _docs, _metas = data["ids"], data["documents"], data["metadatas"]
    _bm25 = BM25Okapi([tokenize(t) for t in _docs])


def bm25_search(query: str, k: int = 20):
    if _bm25 is None:
        build_bm25()
    scores = _bm25.get_scores(tokenize(query))               # one score per chunk
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    best = [i for i in ranked if scores[i] > 0]    # keep only chunks that share a word with the query
    return [
        {"id": _ids[i], "text": _docs[i], "meta": _metas[i], "score": float(scores[i])}
        for i in best
    ]


# ---------- 3. merge the two lists with RRF ----------
def rrf(result_lists, k: int = 60):
    fused = {}
    for results in result_lists:
        for position, chunk in enumerate(results):            # position starts at 0
            entry = fused.setdefault(chunk["id"], {**chunk, "rrf": 0.0})
            entry["rrf"] += 1 / (k + position + 1)            # +1 so the first place is "1"
    return sorted(fused.values(), key=lambda c: c["rrf"], reverse=True)


def hybrid_search(query: str, k: int = 20):
    merged = rrf([vector_search(query, k), bm25_search(query, k)])
    return merged[:k]
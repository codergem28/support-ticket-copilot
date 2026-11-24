# app/rerank.py
from sentence_transformers import CrossEncoder

# a small model that reads (question, page) pairs; it downloads the first time you use it
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, chunks: list, top: int = 5):
    pairs = [(query, c["text"]) for c in chunks]      # one pair per candidate
    scores = cross_encoder.predict(pairs)             # one score per pair
    for chunk, score in zip(chunks, scores):
        chunk["rerank"] = float(score)                # save the score on the chunk
    return sorted(chunks, key=lambda c: c["rerank"], reverse=True)[:top]
# -> usage: python rerank_demo.py "your question"
import sys

from app.rerank import rerank
from app.retrieve import hybrid_search

query = sys.argv[1]
candidates = hybrid_search(query, 20)

print("\nBEFORE reranking (hybrid order):")
for i, c in enumerate(candidates[:5], start=1):
    print(f"{i}. {c['meta']['doc_id']:<30} | {c['text'][:50]!r}")

best = rerank(query, candidates, top=5)

print("\nAFTER reranking (best 5):")
for i, c in enumerate(best, start=1):
    print(f"{i}. score {c['rerank']:>7.2f} | {c['meta']['doc_id']:<28} | {c['text'][:50]!r}")
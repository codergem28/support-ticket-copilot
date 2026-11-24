from app.cache import cache, cache_lookup, cache_put

cache_put("How do I declare a query parameter?", {"answer": "demo answer", "action": "auto_reply"})

tests = [
    "How do I declare a query parameter?",           # identical
    "How can I define a query parameter?",           # same meaning, other words
    "where do I add query parameters to my route",   # same meaning, more different
    "How do I declare a path parameter?",            # different answer, similar words
    "How do I get a refund?",                        # unrelated
]
for q in tests:
    sim, _ = cache_lookup(q)
    print(f"{sim:.3f}  {q}")

cache.delete(ids=cache.get()["ids"]) 
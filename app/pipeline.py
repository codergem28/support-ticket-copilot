import json
import sys
import time

from app.answer import generate
from app.cache import cache_get, cache_put
from app.llm import CHEAP, STRONG
from app.rerank import rerank
from app.retrieve import hybrid_search

# ---- the three dials. These are STARTING values; Day 3's eval will tune them. ----
EASY_RERANK = 5.0   # top rerank score at or above this = easy question -> cheap model
MIN_RERANK = 0.0    # top rerank score below this = docs probably don't cover it -> human
MIN_CONF = 0.7      # the AI's own confidence must be at least this to auto-reply


def as_float(value):
    # the AI sometimes returns the confidence as text like "0.9"; be forgiving
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def handle_ticket(ticket_id: str, question: str):
    start = time.perf_counter()

    # 1. cache
    hit = cache_get(question)
    if hit:
        return {**hit, "source": "cache", "cost": 0.0, "latency": time.perf_counter() - start}

    # 2. find and rerank pages
    chunks = rerank(question, hybrid_search(question, 20), top=5)
    top_score = chunks[0]["rerank"]
    chunk_ids = [c["id"] for c in chunks]

    # 3. nothing relevant -> human, and we never call the AI
    if top_score < MIN_RERANK:
        return {"action": "escalate", "reason": "no_relevant_docs", "top_score": top_score,
                "chunks": chunk_ids, "source": "gate", "cost": 0.0,
                "latency": time.perf_counter() - start}

    # 4. choose the model
    model = CHEAP if top_score >= EASY_RERANK else STRONG
    data, usage = generate(question, chunks, model)
    usages = [usage]

    # 5. cheap model unsure -> try once with the strong model
    if model == CHEAP and as_float(data.get("confidence")) < MIN_CONF:
        data, usage = generate(question, chunks, STRONG)
        usages.append(usage)

    # 6. the gate
    confidence = as_float(data.get("confidence"))
    has_answer = bool(data.get("answer"))
    has_citations = bool(data.get("citations"))
    ok = has_answer and has_citations and confidence >= MIN_CONF

    result = {
        "action": "auto_reply" if ok else "escalate",
        "reason": "ok" if ok else "low_confidence_or_no_citation",
        "answer": data.get("answer", ""),
        "citations": data.get("citations", []),
        "confidence": confidence,
        "top_score": top_score,
        "chunks": chunk_ids,
        "source": usages[-1]["model"],
        "cost": sum(u["cost"] for u in usages),
        "latency": time.perf_counter() - start,
    }
    if ok:
        cache_put(question, result)     # only good, safe answers are remembered
    return result


if __name__ == "__main__":
    out = handle_ticket("test-1", sys.argv[1])
    print(json.dumps(out, indent=2))
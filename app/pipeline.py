# app/pipeline.py
import json
import os
import sys
import time

from app.answer import generate
from app.cache import cache_get, cache_put
from app.llm import CHEAP, STRONG
from app.obs import span, trace
from app.rerank import rerank
from app.retrieve import hybrid_search, vector_search

# ---- feature switches: set to "0" in the environment to turn a feature off ----
CFG = {name: os.getenv(name, "1") == "1"
       for name in ("USE_HYBRID", "USE_RERANK", "USE_CACHE", "USE_ROUTING")}

# ---- the dials (tuned on the dev split, see eval/) ----
EASY_RERANK = 5.0
# Reranker scores are logits, not percentages: a correct page on a long, chit-chatty
# question routinely scores negative. Over the 100 golden rows the best page scored
# as low as -8.19 on answerable questions, while out-of-scope questions topped out at
# -0.50. -6.0 keeps 58 of 59 answerable dev rows in play (0.0 dropped 12 of them) and
# still blocks the clearly-irrelevant ones; the confidence/citation gate below is the
# real safety net for the out-of-scope rows that score above it.
MIN_RERANK = -6.0
MIN_CONF = 0.7


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def handle_ticket(ticket_id: str, question: str):
    with trace("ticket", ticket_id=ticket_id) as t:
        result = _handle(ticket_id, question)
        t.update(action=result["action"], reason=result.get("reason"),
                 source=result.get("source"), cost=result.get("cost", 0.0),
                 top_score=result.get("top_score"))
        return result


def _handle(ticket_id: str, question: str):
    start = time.perf_counter()

    # 1. cache
    with span("cache") as s:
        hit = cache_get(question) if CFG["USE_CACHE"] else None
        s["hit"] = bool(hit)
    if hit:
        return {**hit, "source": "cache", "cost": 0.0, "latency": time.perf_counter() - start}

    # 2. find pages
    with span("retrieve") as s:
        candidates = hybrid_search(question, 20) if CFG["USE_HYBRID"] else vector_search(question, 20)
        s["candidates"] = len(candidates)
    if CFG["USE_RERANK"]:
        with span("rerank") as s:
            chunks = rerank(question, candidates, top=5)
            top_score = chunks[0]["rerank"]
            s["top_score"] = round(top_score, 2)
    else:
        chunks, top_score = candidates[:5], None
    chunk_ids = [c["id"] for c in chunks]

    # 3. nothing relevant -> human, no AI call
    if top_score is not None and top_score < MIN_RERANK:
        return {"action": "escalate", "reason": "no_relevant_docs", "top_score": top_score,
                "chunks": chunk_ids, "source": "gate", "cost": 0.0,
                "latency": time.perf_counter() - start}

    # 4. choose the model
    easy = CFG["USE_ROUTING"] and top_score is not None and top_score >= EASY_RERANK
    model = CHEAP if easy else STRONG
    with span("llm") as s:
        data, usage = generate(question, chunks, model)
        s.update(model=usage["model"], tokens_in=usage["in"], tokens_out=usage["out"], cost=usage["cost"])
    usages = [usage]

    # 5. cheap model unsure -> strong model once
    if model == CHEAP and as_float(data.get("confidence")) < MIN_CONF:
        with span("llm_retry") as s:
            data, usage = generate(question, chunks, STRONG)
            s.update(model=usage["model"], tokens_in=usage["in"], tokens_out=usage["out"], cost=usage["cost"])
        usages.append(usage)

    # 6. the gate
    confidence = as_float(data.get("confidence"))
    ok = bool(data.get("answer")) and bool(data.get("citations")) and confidence >= MIN_CONF

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
    if ok and CFG["USE_CACHE"]:
        cache_put(question, result)
    return result


if __name__ == "__main__":
    print(json.dumps(handle_ticket("test-1", sys.argv[1]), indent=2))
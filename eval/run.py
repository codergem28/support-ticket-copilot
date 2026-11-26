# eval/run.py
# usage: python -m eval.run --name run1 --split dev [--fraction 0.5] [--sleep 1.5]
import argparse
import hashlib
import json
import math
import os
import time

from app.cache import cache
from app.pipeline import CFG, handle_ticket
from eval.judge import judge


def percentile(values, p):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def avg(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def doc_of(chunk_id):
    return chunk_id.split("::")[0]                     # chunk ids look like "docname::hash"


def keep(row, fraction):
    # deterministic sample; a paraphrase is kept exactly when its original is kept
    key = row.get("paraphrase_of", row["id"])
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 100 < fraction * 100


def load_rows(split, fraction):
    path = "eval/golden.jsonl" if os.path.exists("eval/golden.jsonl") else "eval/golden_draft.jsonl"
    print("using", path)
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return [r for r in rows if r["split"] == split and keep(r, fraction)]


def main(name, split, fraction, pause):
    rows = load_rows(split, fraction)
    old = cache.get()["ids"]
    if old:
        cache.delete(ids=old)                           # every run starts with an empty cache

    per_row, errors = [], 0
    for i, r in enumerate(rows, start=1):
        try:
            res = handle_ticket(r["id"], r["question"])
        except Exception as e:
            errors += 1
            print("error on", r["id"], "|", e)
            continue
        docs = [doc_of(c) for c in res.get("chunks", [])]
        entry = {
            "id": r["id"], "question": r["question"], "should_answer": r["should_answer"],
            "action": res["action"], "source": res.get("source"), "cost": res.get("cost", 0.0),
            "latency": res["latency"], "answer": res.get("answer", ""),
            "rank": next((k + 1 for k, d in enumerate(docs) if d in r["expected_docs"]), None),
        }
        if res["action"] == "auto_reply":
            j = judge(r["question"], res["chunks"], res["answer"], r["reference"])
            entry.update(faithful=j["faithful"], correct=j["correct"], judge_reason=j["reason"])
        per_row.append(entry)
        print(f"[{i}/{len(rows)}] {r['id']} -> {entry['action']} ({entry['source']})")
        time.sleep(pause)

    answerable = [e for e in per_row if e["should_answer"]]
    unanswerable = [e for e in per_row if not e["should_answer"]]
    autos = [e for e in per_row if e["action"] == "auto_reply"]
    bad_autos = [e for e in autos if (not e["should_answer"]) or not (e["faithful"] and e["correct"])]

    summary = {
        "name": name, "split": split, "fraction": fraction, "rows": len(per_row), "errors": errors,
        "cfg": CFG,
        "hit_rate@5": avg(e["rank"] is not None for e in answerable),
        "mrr": avg(1 / e["rank"] if e["rank"] else 0 for e in answerable),
        "faithfulness": avg(e["faithful"] for e in autos),
        "correctness": avg(e["correct"] for e in autos if e["should_answer"]),
        "deflection_rate": len(autos) / len(per_row) if per_row else None,
        "wrong_auto_reply_rate": len(bad_autos) / len(autos) if autos else 0.0,
        "correct_escalation_rate": avg(e["action"] == "escalate" for e in unanswerable),
        "cost_per_ticket_usd": avg(e["cost"] for e in per_row),
        "total_cost_usd": sum(e["cost"] for e in per_row),
        "p50_s": percentile([e["latency"] for e in per_row], 0.50) if per_row else None,
        "p95_s": percentile([e["latency"] for e in per_row], 0.95) if per_row else None,
    }

    os.makedirs("eval/results", exist_ok=True)
    json.dump(summary, open(f"eval/results/{name}.json", "w", encoding="utf-8"), indent=2)
    with open(f"eval/results/{name}.rows.jsonl", "w", encoding="utf-8") as f:
        for e in per_row:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--sleep", type=float, default=1.5)
    a = ap.parse_args()
    main(a.name, a.split, a.fraction, a.sleep)
import hashlib
import json
import pathlib
import time

from app.llm import STRONG, chat_json

MAX_DOCS = 40

SYSTEM_Q = """You write test questions for a customer-support bot.
You are given ONE help article. Write 2 realistic questions a confused customer might ask that this article answers.
Rules:
- Write like a customer who has NOT read the docs: everyday words, never copy phrases from the article.
- Each question must be answerable from this article alone.
- Give a short reference answer (1 or 2 sentences) based only on the article.
- Give one paraphrase of each question (same meaning, different words).
Return JSON: {"items": [{"question": str, "reference": str, "paraphrase": str}]}"""

SYSTEM_OOS = """You write test questions for a customer-support bot for a software library's documentation.
Write customer questions that the technical documentation can NOT answer: billing, refunds, pricing,
account access, outages, company policies, legal terms, sales requests.
Return JSON: {"questions": [str]}"""


def split_for(doc_id: str) -> str:
    # same document always lands in the same split: about 70% dev, 30% test
    return "dev" if int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % 10 < 7 else "test"


def main():
    rows, n = [], 0
    files = sorted(pathlib.Path("docs").glob("*.md"))[:MAX_DOCS]
    for index, path in enumerate(files):
        text = path.read_text(encoding="utf-8")[:6000]       # keep prompts small
        try:
            data, _ = chat_json(STRONG, SYSTEM_Q, f"ARTICLE:\n{text}")
        except Exception as e:
            print("skipped", path.stem, "|", e)
            continue
        split = split_for(path.stem)
        for j, item in enumerate(data.get("items", [])[:2]):
            if not isinstance(item, dict) or not item.get("question") or not item.get("reference"):
                continue
            n += 1
            qid = f"q{n:03d}"
            rows.append({"id": qid, "question": item["question"], "expected_docs": [path.stem],
                         "reference": item["reference"], "should_answer": True, "split": split})
            # a paraphrase for the first question of every second document
            if j == 0 and index % 2 == 0 and item.get("paraphrase"):
                rows.append({"id": qid + "p", "question": item["paraphrase"], "expected_docs": [path.stem],
                             "reference": item["reference"], "should_answer": True, "split": split,
                             "paraphrase_of": qid})
        print("done", path.stem)
        time.sleep(4)                                        # be gentle with the free tier

    oos, _ = chat_json(STRONG, SYSTEM_OOS, "Write 24 questions.")
    for i, q in enumerate(oos.get("questions", [])):
        rows.append({"id": f"o{i + 1:03d}", "question": q, "expected_docs": [], "reference": "",
                     "should_answer": False, "split": "dev" if i % 10 < 7 else "test"})

    rows.sort(key=lambda r: "paraphrase_of" in r)            # originals first, paraphrases last
    with open("eval/golden_draft.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to eval/golden_draft.jsonl")


if __name__ == "__main__":
    main()
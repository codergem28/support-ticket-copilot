import os

from dotenv import load_dotenv

from app.llm import STRONG, chat_json
from app.store import col

load_dotenv()
JUDGE_MODEL = os.getenv("MODEL_JUDGE") or STRONG

JUDGE_SYSTEM = """You are a strict grader of customer-support answers. You are given a question, the help pages the answer was allowed to use (CONTEXT), the ANSWER to grade, and a REFERENCE answer.

Score two things separately, each 0 or 1:
- faithful: 1 only if EVERY factual claim in ANSWER is supported by CONTEXT. Invented parameter names, defaults, versions or behaviour mean 0. A claim being true in real life does not help: CONTEXT is the only allowed source, so a correct fact that is not in CONTEXT is still 0.
- correct: 1 only if ANSWER agrees in meaning with REFERENCE. Missing the key point, contradicting it, or answering a different question means 0. If REFERENCE is "(none - the correct behaviour is to decline)", then correct is 1 only if ANSWER declines or hands off to a human, and 0 if it gives a technical answer.

Do not reward length or confidence. Do not use your own knowledge of the subject. Keep reason to one sentence.
Return JSON: {"faithful": 0 or 1, "correct": 0 or 1, "reason": str}"""


def verdict(value) -> int:
    """Be tolerant of True/"1"/"yes", strict about everything else."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value == 1)
    return int(str(value).strip().lower() in ("1", "true", "yes"))


def judge(question: str, chunk_ids: list, answer: str, reference: str):
    """Grade one answer against the pages it was based on.

    Pass the retrieved chunk ids (pipeline result["chunks"]), not just the citations:
    the brief is faithfulness to the pages the answer was based on.
    """
    if chunk_ids:
        got = col.get(ids=list(chunk_ids))
        docs = dict(zip(got["ids"], got["documents"]))
        context = "\n\n".join(f"[{i}]\n{docs[i]}" for i in chunk_ids if i in docs)
    else:
        context = ""
    context = context or "(no pages were provided)"

    user = (f"QUESTION: {question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer or '(empty)'}\n\n"
            f"REFERENCE: {reference or '(none - the correct behaviour is to decline)'}")

    data = {}
    for _ in range(2):                      # one retry: a garbled reply must not look like a real 0/0
        try:
            data, _ = chat_json(JUDGE_MODEL, JUDGE_SYSTEM, user)
        except Exception as e:              # network/provider failure: report, do not score
            print("judge call failed:", e)
            continue
        if "faithful" in data and "correct" in data:
            return {"faithful": verdict(data["faithful"]), "correct": verdict(data["correct"]),
                    "reason": str(data.get("reason", "")), "ok": True}
    print("judge returned no verdict:", data)
    return {"faithful": None, "correct": None, "reason": "judge_error", "ok": False}

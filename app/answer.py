# app/answer.py
import json
import re
import sys

from app.llm import CHEAP, STRONG, chat_json
from app.retrieve import vector_search

SYSTEM = """You are a support assistant. Answer the customer's question using ONLY the context.
Rules:
- citations must list every chunk id you used, copied exactly from the brackets in the context.
- An answer with an empty citations list is not allowed. If you cannot cite a chunk id for a
  statement, do not make the statement.
- If the context does not contain the answer, return {"answer": "", "citations": [], "confidence": 0.0}.
- confidence is a number from 0 to 1 for how fully the context answers the question.
Return JSON only, in this exact shape: {"answer": str, "citations": [str], "confidence": float}"""


def _raw_citations(citations):
    """Flatten whatever shape the AI returned into a list of bare strings.

    Seen in practice: ["id"], "['id']" (whole list as one string), "[id]",
    '`id`', "id, id2", and a list nested inside a list. All of these are the
    same citation and none of them should be thrown away.
    """
    if isinstance(citations, (str, int, float)):
        citations = [citations]
    if not isinstance(citations, (list, tuple)):
        return []
    out = []
    for item in citations:
        if isinstance(item, (list, tuple)):
            out.extend(_raw_citations(item))
            continue
        text = str(item).strip().strip("[]{}").strip()
        for part in re.split(r"[,;\n]", text):
            part = part.strip().strip("'\"`[]{}(),;").strip()
            if part:
                out.append(part)
    return out


def _resolve_citations(candidates, chunks):
    """Keep only citations that point at a page we actually sent.

    Matches the full chunk id first. If the model gave just the page name
    ("response-model" or "response-model::wronghash") we accept it only when
    one chunk of that page is in the context, so we can't mislabel a page.
    """
    by_id = {c["id"]: c["id"] for c in chunks}
    stems = {}
    for c in chunks:
        stems.setdefault(c["id"].split("::")[0], []).append(c["id"])

    out = []
    for cand in candidates:
        if cand in by_id:
            out.append(by_id[cand])
            continue
        same_page = stems.get(cand.split("::")[0], [])
        if len(same_page) == 1:
            out.append(same_page[0])
    return list(dict.fromkeys(out))


REPAIR = """You match answers to the pages they came from.
You get a question, an answer, and the pages the answer was written from.
List the ids of the pages that directly support statements in the answer.
Use ONLY ids from the pages given. If nothing in the pages supports the answer, return {"citations": []}.
Return JSON only: {"citations": [str]}"""


def _repair_citations(question, chunks, answer, model):
    """The model sometimes answers without filling in citations. Ask once, cheaply,
    which of the pages it was given back up the answer. Ids are still validated
    against the context afterwards, so this can only narrow, never invent."""
    listing = "\n\n".join(f"[{c['id']}]\n{c['text'][:600]}" for c in chunks)
    user = f"Question: {question}\n\nAnswer:\n{answer}\n\nPages:\n{listing}"
    try:
        data, usage = chat_json(model, REPAIR, user)
    except Exception:
        return [], None
    return _resolve_citations(_raw_citations(data.get("citations")), chunks), usage


def generate(question: str, chunks: list, model: str):
    """Ask the AI to answer from the given chunks. Returns (data, usage)."""
    context = "\n\n".join(f"[{c['id']}]\n{c['text']}" for c in chunks)
    user = f"Context:\n{context}\n\nQuestion: {question}"
    data, usage = chat_json(model, SYSTEM, user)

    # never trust the AI's citations: keep only ids that really were in the context,
    # but accept every reasonable way of writing them down.
    data["citations"] = _resolve_citations(_raw_citations(data.get("citations")), chunks)

    # next: the model put the ids in the prose instead of in the field.
    # Only exact ids count here, so we never invent a source.
    if not data["citations"] and data.get("answer"):
        in_text = [c["id"] for c in chunks if c["id"] in data["answer"]]
        data["citations"] = list(dict.fromkeys(in_text))

    # last resort: an answer exists but cites nothing. Ask which pages support it.
    if not data["citations"] and data.get("answer"):
        citations, repair_usage = _repair_citations(question, chunks, data["answer"], model)
        data["citations"] = citations
        if repair_usage:
            usage = {**usage, "in": usage["in"] + repair_usage["in"],
                     "out": usage["out"] + repair_usage["out"],
                     "cost": usage["cost"] + repair_usage["cost"],
                     "latency": usage["latency"] + repair_usage["latency"]}
    return data, usage


def answer(question: str, model: str | None = None, k: int = 5):
    model = model or CHEAP
    chunks = vector_search(question, k)
    data, usage = generate(question, chunks, model)
    return {**data, "chunks": chunks, "usage": usage}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--strong"]
    model = STRONG if "--strong" in sys.argv else None
    r = answer(args[0], model=model)
    print(json.dumps({k: v for k, v in r.items() if k != "chunks"}, indent=2))
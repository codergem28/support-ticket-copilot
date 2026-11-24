# app/answer.py
import json
import sys

from app.llm import CHEAP, STRONG, chat_json
from app.retrieve import vector_search

SYSTEM = """You are a support assistant. Answer the customer's question using ONLY the context.
Rules:
- If the context does not contain the answer, return {"answer": "", "citations": [], "confidence": 0.0}.
- Cite the chunk ids you used, exactly as written in the brackets.
- confidence is a number from 0 to 1 for how fully the context answers the question.
Return JSON: {"answer": str, "citations": [str], "confidence": float}"""


def generate(question: str, chunks: list, model: str):
    """Ask the AI to answer from the given chunks. Returns (data, usage)."""
    context = "\n\n".join(f"[{c['id']}]\n{c['text']}" for c in chunks)
    user = f"Context:\n{context}\n\nQuestion: {question}"
    data, usage = chat_json(model, SYSTEM, user)

    # never trust the AI's citations: keep only ids that really were in the context.
    # The model often echoes them with the brackets it saw ([id]), so normalise first.
    valid = {c["id"] for c in chunks}
    cleaned = [str(c).strip().strip("[]").strip() for c in data.get("citations", [])]
    data["citations"] = list(dict.fromkeys(c for c in cleaned if c in valid))
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
import json
import os
import sys

from app.llm import chat_json
from app.retrieve import vector_search

SYSTEM = """You are a support assistant. Answer the customer's question using ONLY the context.
Rules:
- If the context does not contain the answer, return {"answer": "", "citations": [], "confidence": 0.0}.
- Cite the chunk ids you used, exactly as written in the brackets.
- confidence is a number from 0 to 1 for how fully the context answers the question.
Return JSON: {"answer": str, "citations": [str], "confidence": float}"""


def answer(question: str, model: str | None = None, k: int = 5):
    model = model or os.getenv("MODEL_CHEAP")
    chunks = vector_search(question, k)
    context = "\n\n".join(f"[{c['id']}]\n{c['text']}" for c in chunks)
    user = f"Context:\n{context}\n\nQuestion: {question}"
    data, usage = chat_json(model, SYSTEM, user)

    # never trust the AI's citations: keep only ids that really were retrieved
    valid = {c["id"] for c in chunks}
    data["citations"] = [c for c in data.get("citations", []) if c in valid]
    return {**data, "chunks": chunks, "usage": usage}


if __name__ == "__main__":
    r = answer(sys.argv[1])
    print(json.dumps({k: v for k, v in r.items() if k != "chunks"}, indent=2))
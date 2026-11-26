import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.environ["LLM_API_KEY"])

CHEAP = os.getenv("MODEL_CHEAP", "cheap")
STRONG = os.getenv("MODEL_STRONG", "strong")
JUDGE = os.getenv("MODEL_JUDGE", STRONG)   # grade with a different model family if you have one

# USD per 1 million tokens: (input price, output price).
# Look up the real prices for YOUR model. Free tiers can stay at 0.
PRICES = {
    CHEAP: (0.075, 0.30),    # openai/gpt-oss-20b
    STRONG: (0.15, 0.60),    # openai/gpt-oss-120b
}
PRICES.setdefault(JUDGE, (0.80, 4.00))   # qwen/qwen3.8-27b, the only non-gpt-oss family on this key


def chat_json(model, system, user):
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},  # delete this line if your provider rejects it
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    latency = time.perf_counter() - t0
    price_in, price_out = PRICES.get(model, (0, 0))
    u = resp.usage
    cost = (u.prompt_tokens * price_in + u.completion_tokens * price_out) / 1_000_000
    try:
        data = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        data = {"answer": "", "citations": [], "confidence": 0.0}
    usage = {"model": model, "in": u.prompt_tokens, "out": u.completion_tokens,
             "cost": cost, "latency": latency}
    return data, usage
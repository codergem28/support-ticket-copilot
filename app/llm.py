import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.environ["LLM_API_KEY"])

CHEAP = os.getenv("MODEL_CHEAP", "cheap")
STRONG = os.getenv("MODEL_STRONG", "strong")

# USD per 1 million tokens: (input price, output price).
# Look up the real prices for YOUR model. Free tiers can stay at 0.
PRICES = {
    CHEAP: (0.0, 0.0),
    STRONG: (0.0, 0.0),
}


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
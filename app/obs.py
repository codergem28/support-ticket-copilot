import contextlib
import contextvars
import json
import os
import time
import uuid

TRACE_FILE = os.getenv("TRACE_FILE", "data/traces.jsonl")
_current = contextvars.ContextVar("trace", default=None)


@contextlib.contextmanager
def trace(name, **attrs):
    """One trace = one ticket. Written to the file when the ticket finishes."""
    t = {"trace_id": uuid.uuid4().hex, "name": name, "ts": time.time(), "spans": [], **attrs}
    token, start = _current.set(t), time.perf_counter()
    try:
        yield t
    finally:
        t["latency"] = time.perf_counter() - start
        _current.reset(token)
        os.makedirs(os.path.dirname(TRACE_FILE), exist_ok=True)
        with open(TRACE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


@contextlib.contextmanager
def span(name, **attrs):
    """One span = one step inside the current trace."""
    s, start = {"name": name, **attrs}, time.perf_counter()
    try:
        yield s
    finally:
        s["ms"] = round((time.perf_counter() - start) * 1000, 1)
        t = _current.get()
        if t is not None:
            t["spans"].append(s)
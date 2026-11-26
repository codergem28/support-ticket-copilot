# Ticket Copilot — RAG support bot for the FastAPI docs

A support bot that answers questions from the FastAPI documentation, and escalates
to a human when the docs don't cover the question.

Three long-running containers plus a one-shot ingest step, one command.

## Deploy

Prerequisites: Docker, and an LLM API key (Groq by default).

```bash
cp .env.example .env      # then put your key in LLM_API_KEY
docker compose up --build
```

Then open http://localhost:8000/docs.

Before deploying, check that nothing else is listening on port 8000 — a leftover local
`uvicorn` will silently answer some requests with its own (possibly stale) code:
`netstat -ano | grep :8000` should show only the Docker proxy.

What happens on `up`:

1. **redis** starts and reports healthy.
2. **ingest** runs once, embeds `./docs/*.md` into the Chroma volume, and exits.
   Re-running is cheap and idempotent: only new or changed chunks are embedded.
3. **worker** starts only after ingest succeeds, loads the models, and waits for tickets.
4. **api** serves on port 8000 with a `/health` check wired to Redis.

### Services

| Service | Role |
|---|---|
| `api` | FastAPI: `POST /tickets` queues a ticket, `GET /tickets/{id}` returns the result |
| `worker` | reads the Redis stream, runs the pipeline, writes the result back |
| `ingest` | one-shot: chunks + embeds `docs/` into the vector store |
| `redis` | queue (`ticket.created`), results, dead-letter queue |

### Using it

```bash
# post a ticket
curl -s -X POST http://localhost:8000/tickets \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I create a FastAPI route?"}'
# -> {"ticket_id":"998f04ea..."}

# fetch the result ({"status":"pending"} until the worker finishes)
curl -s http://localhost:8000/tickets/998f04ea...
```

The result contains `action` (`auto_reply` or `escalate`), `reason`, `answer`,
`citations`, `confidence`, `top_score`, `source` (model or `gate`/`cache`),
`cost`, `latency` and `queue_wait`.

If `DEMO_KEY` is set in `.env`, every endpoint except `/health` requires an
`X-API-Key` header. Unset means open, which is fine for local use.

## How it answers

1. Retrieve: vector search over doc chunks, merged with BM25 by RRF (`USE_HYBRID`).
2. Rerank: a cross-encoder rescores the top candidates (`USE_RERANK`).
3. Gate 1: if the best rerank score is below `MIN_RERANK`, escalate without calling the model.
4. Route: score >= `EASY_RERANK` goes to the cheap model, otherwise the strong one (`USE_ROUTING`).
5. Gate 2: auto-reply only with an answer, at least one valid citation, and confidence >= `MIN_CONF`.
6. Cache: successful replies are stored and reused for similar questions (`USE_CACHE`).

Dials live at the top of `app/pipeline.py` and were tuned on `eval/golden.jsonl`:

- `MIN_RERANK = -6.0` — reranker scores are logits, not percentages. Correct pages on
  long conversational questions routinely score negative (as low as −8.19 across the
  golden set), while out-of-scope questions topped out at −0.50. `0.0` was rejecting
  12 of 59 answerable dev questions.
- `EASY_RERANK = 5.0` — the cheap model handles clearly-easy questions.
- `MIN_CONF = 0.7`.

Set any switch to `0` in `.env` to disable that feature.

## Limitations

- **Do not scale the worker horizontally with this setup.** Embedded Chroma
  (`data/chroma`, the `chroma` volume) is built for a single process: `PersistentClient`
  is not safe for concurrent writers, so two workers writing the cache and index at the
  same time can corrupt the store or raise errors. The fix is to swap embedded Chroma
  for a **Qdrant server** (or the Chroma HTTP server) and point the app at it, then run
  as many workers as you need. Until then, run exactly one worker.
- **`q012`-style retrieval misses**: some questions name a doc page that never reaches
  the top 5 (e.g. `body-updates`). The reranker gate can no longer block those, so the
  model answers from the wrong pages and the confidence gate has to catch it. Worth
  revisiting chunking rather than the threshold.
- **Free-tier LLM limits**: on Groq's free tier, `gpt-oss-120b` allows 200k tokens/day
  and 8k tokens/minute. A 100-row eval run is ~340k tokens, so full ablation runs must
  be split across days or sampled (see `eval/`).
- Model downloads happen at image build time, so the first build is slow and needs network access.
- No authentication on Redis; it is not published outside the compose network.
- If Redis is unreachable the API answers `503` with the reason, not a bare `500`.
- The worker reclaims tickets left unacknowledged by a crashed worker after 60s, so an
  interrupted ticket is retried rather than lost; after 3 attempts it goes to `ticket.dlq`.

## Layout

```
app/api.py        FastAPI app
app/worker.py     Redis stream consumer, retries + dead-letter queue
app/pipeline.py   retrieval -> rerank -> route -> answer -> gate
app/retrieve.py   vector search, BM25, RRF merge
app/rerank.py     cross-encoder reranking
app/answer.py     prompt, citation validation, repair pass
app/ingest.py     docs -> chunks -> embeddings
app/store.py      Chroma collection + embedding model
eval/             golden set, judge, score dumps
docs/             FastAPI documentation source
```

import json
import os
import time

import redis

from app.pipeline import handle_ticket

BLOCK_MS = 5000            # how long XREADGROUP waits for new tickets

# socket_timeout must outlast the blocking read: redis-py 8.x defaults it to 5s,
# which equals BLOCK_MS and makes every idle wait die with TimeoutError.
r = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), decode_responses=True,
                socket_timeout=BLOCK_MS / 1000 + 5)

STREAM = "ticket.created"
GROUP = "workers"
DLQ = "ticket.dlq"          # the "dead-letter queue": where tickets go when they keep failing
MAX_ATTEMPTS = 3
CLAIM_IDLE_MS = 60_000      # a ticket unacked this long belongs to a worker that died
ME = f"worker-{os.getpid()}"


def setup():
    # create the shared group once; "BUSYGROUP" just means it already exists
    try:
        r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


def process(fields):
    ticket_id = fields["ticket_id"]
    if r.exists(f"result:{ticket_id}"):            # already handled -> do nothing (idempotent)
        return
    queue_wait = time.time() - float(fields["ts"])
    result = handle_ticket(ticket_id, fields["question"])
    result["queue_wait"] = queue_wait
    r.set(f"result:{ticket_id}", json.dumps(result))
    event = "ticket.answered" if result["action"] == "auto_reply" else "ticket.escalated"
    r.xadd(event, {"ticket_id": ticket_id})        # announce the outcome as a new event


def handle_message(message_id, fields):
    attempt = int(fields.get("attempt", 0))
    try:
        process(fields)
        print("done:", fields["ticket_id"])
    except Exception as e:
        print("failed:", fields.get("ticket_id"), "|", e)
        if attempt + 1 >= MAX_ATTEMPTS:
            r.xadd(DLQ, {**fields, "error": str(e)})            # give up, park it
        else:
            time.sleep(2 ** attempt)                            # wait 1s, then 2s
            r.xadd(STREAM, {**fields, "attempt": attempt + 1})  # try again later
    finally:
        r.xack(STREAM, GROUP, message_id)                       # tell Redis we're done with this copy


def claim_stale():
    """Pick up tickets a previous worker read but never acknowledged (it crashed)."""
    start = "0-0"
    while True:
        try:
            start, messages, _ = r.xautoclaim(STREAM, GROUP, ME, CLAIM_IDLE_MS,
                                             start_id=start, count=50)
        except redis.ResponseError:
            return
        ready = [(mid, f) for mid, f in messages if f]
        for message_id, fields in ready:
            print("reclaiming:", fields.get("ticket_id"))
            handle_message(message_id, fields)
        if not ready or start in ("0-0", 0):
            return


def run():
    setup()
    claim_stale()
    print(f"{ME} ready, waiting for tickets...")
    while True:
        try:
            batches = r.xreadgroup(GROUP, ME, {STREAM: ">"}, count=5, block=BLOCK_MS) or []
        except redis.RedisError as e:
            # Redis restarted or the connection dropped: stay alive instead of
            # crashing the container and leaving the group half-registered.
            print("redis unavailable, retrying:", e)
            time.sleep(2)
            try:
                setup()
            except redis.RedisError:
                pass
            continue
        for _, messages in batches:
            for message_id, fields in messages:
                handle_message(message_id, fields)


if __name__ == "__main__":
    run()

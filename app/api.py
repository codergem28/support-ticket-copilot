# app/api.py
import json
import os
import time
import uuid

import redis
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
r = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), decode_responses=True)


class Ticket(BaseModel):
    question: str


@app.post("/tickets", status_code=202)
def create_ticket(ticket: Ticket):
    ticket_id = uuid.uuid4().hex
    # put the ticket on the waiting line and answer immediately; no AI work happens here
    r.xadd("ticket.created", {"ticket_id": ticket_id, "question": ticket.question, "ts": time.time()})
    return {"ticket_id": ticket_id}


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str):
    value = r.get(f"result:{ticket_id}")
    return json.loads(value) if value else {"status": "pending"}
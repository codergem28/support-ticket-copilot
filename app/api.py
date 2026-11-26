# app/api.py
import json
import os
import time
import uuid

import redis
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI()
r = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), decode_responses=True)


class Ticket(BaseModel):
    question: str

def check_key(x_api_key: str = Header(default="")):
    """Only enforced when DEMO_KEY is set; unset means open (local dev)."""
    secret = os.getenv("DEMO_KEY")
    if secret and x_api_key != secret:
        raise HTTPException(status_code=401, detail="missing or wrong API key")


@app.get("/health")
def health():
    try:
        r.ping()
    except redis.RedisError as e:
        raise HTTPException(status_code=503, detail=f"redis unavailable: {e}")
    return {"status": "ok"}


@app.post("/tickets", status_code=202, dependencies=[Depends(check_key)])
def create_ticket(ticket: Ticket):
    ticket_id = uuid.uuid4().hex
    # put the ticket on the waiting line and answer immediately; no AI work happens here
    try:
        r.xadd("ticket.created", {"ticket_id": ticket_id, "question": ticket.question, "ts": time.time()})
    except redis.RedisError as e:
        raise HTTPException(status_code=503, detail=f"queue unavailable: {e}")
    return {"ticket_id": ticket_id}


@app.get("/tickets/{ticket_id}", dependencies=[Depends(check_key)])
def get_ticket(ticket_id: str):
    try:
        value = r.get(f"result:{ticket_id}")
    except redis.RedisError as e:
        raise HTTPException(status_code=503, detail=f"store unavailable: {e}")
    return json.loads(value) if value else {"status": "pending"}

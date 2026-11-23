import hashlib
import pathlib
import re

from app.store import col, embed_docs


def split_sections(text):
    # split the file at every markdown heading (#, ##, ###)
    parts = re.split(r"(?m)^(?=#{1,3} )", text)
    return [p.strip() for p in parts if p.strip()]


def chunk(text, max_words=250, overlap=40):
    # cut a section into pieces of about 250 words, overlapping by 40
    words = text.split()
    step = max_words - overlap
    for i in range(0, max(len(words), 1), step):
        piece = words[i:i + max_words]
        if piece:
            yield " ".join(piece)
        if i + max_words >= len(words):
            break


def ingest_file(path: pathlib.Path):
    doc_id = path.stem                      # file name without .md
    text = path.read_text(encoding="utf-8")

    chunks = {}
    for sec in split_sections(text):
        title = sec.splitlines()[0].lstrip("# ").strip()
        for c in chunk(sec):
            h = hashlib.sha256(c.encode()).hexdigest()[:16]   # fingerprint of the text
            cid = f"{doc_id}::{h}"
            chunks[cid] = {"text": c, "meta": {"doc_id": doc_id, "section": title}}

    old_ids = set(col.get(where={"doc_id": doc_id})["ids"])   # what is already stored
    stale = list(old_ids - set(chunks))                       # stored but no longer in the file
    if stale:
        col.delete(ids=stale)

    todo = [cid for cid in chunks if cid not in old_ids]      # only new text needs embedding
    if todo:
        col.upsert(
            ids=todo,
            documents=[chunks[i]["text"] for i in todo],
            metadatas=[chunks[i]["meta"] for i in todo],
            embeddings=embed_docs([chunks[i]["text"] for i in todo]),
        )
    return {"doc": doc_id, "total": len(chunks), "embedded": len(todo), "deleted": len(stale)}


if __name__ == "__main__":
    for p in sorted(pathlib.Path("docs").glob("*.md")):
        print(ingest_file(p))
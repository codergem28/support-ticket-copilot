# app/store.py
import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

model = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path="data/chroma")
col = client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})


def embed_docs(texts):
    return model.encode(texts, normalize_embeddings=True).tolist()


def embed_query(text):
    return model.encode([QUERY_PREFIX + text], normalize_embeddings=True).tolist()[0]
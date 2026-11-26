FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# CPU-only PyTorch keeps the image much smaller than the default build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

# download the two models at build time, so containers start fast
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY app ./app
COPY eval ./eval
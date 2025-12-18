from openai import OpenAI
import os
import glob
import faiss
import pickle
import math

client = OpenAI()

# CONFIG
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"   # example; pick available model in your account
DOCS_DIR = "docs"
INDEX_FILE = "faiss.index"
META_FILE = "faiss_meta.pkl"
CHUNK_SIZE = 800   # characters per chunk
CHUNK_OVERLAP = 100

# Helper: simple chunker
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    start = 0
    chunks = []
    L = len(text)
    while start < L:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = max(start + chunk_size - overlap, end)
    return chunks

# Build or load index
def build_index():
    docs = []
    metadatas = []
    for path in glob.glob(os.path.join(DOCS_DIR, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        for i, chunk in enumerate(chunk_text(text)):
            docs.append(chunk)
            metadatas.append({"source": os.path.basename(path), "chunk": i})

    # get embeddings in batches
    batch_size = 64
    vectors = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        batch_vecs = [r.embedding for r in resp.data]
        vectors.extend(batch_vecs)

    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors, dtype="float32"))
    # persist
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump({"docs": docs, "metas": metadatas}, f)
    print(f"Built index with {len(docs)} passages")
    return index, docs, metadatas

# Load index
def load_index():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        return build_index()
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
    return index, meta["docs"], meta["metas"]

import numpy as np

index, docs, metas = load_index()

# Retrieval + generation
def retrieve(query, k=4):
    q_emb = client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
    D, I = index.search(np.array([q_emb], dtype="float32"), k)
    results = []
    for idx in I[0]:
        results.append({"text": docs[idx], "meta": metas[idx]})
    return results

def create_prompt(query, retrieved):
    ctx = "\n\n---\n\n".join([f"Source: {r['meta']['source']} (chunk {r['meta']['chunk']})\n{r['text']}" for r in retrieved])
    prompt = (
        "You are a helpful assistant. Use the provided context to answer the user.\n\n"
        "Context:\n" + ctx + "\n\n"
        f"User question: {query}\n\n"
        "Answer concisely and cite sources by filename when useful."
    )
    return prompt

def answer(query):
    retrieved = retrieve(query, k=4)
    prompt = create_prompt(query, retrieved)
    resp = client.responses.create(model=LLM_MODEL, input=prompt, max_tokens=600)
    # responses API returns a content structure; adjust if your client variant differs
    text = ""
    # try to get text content
    if hasattr(resp, "output") and isinstance(resp.output, list):
        for item in resp.output:
            if isinstance(item, dict) and "content" in item:
                for c in item["content"]:
                    if c.get("type") == "output_text":
                        text += c.get("text", "")
    else:
        # fallback to top-level text
        text = getattr(resp, "text", "") or resp.output_text if hasattr(resp, "output_text") else ""
    return text.strip()

if __name__ == "__main__":
    # Quick interactive loop
    print("RAG demo. Make sure you have text files in the 'docs' folder.")
    while True:
        q = input("\nEnter question (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            break
        ans = answer(q)
        print("\nAnswer:\n", ans)
# ...existing code...
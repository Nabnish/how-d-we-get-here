import os
import glob
import pickle
import faiss
import numpy as np
from openai import OpenAI

# =========================
# Configuration
# =========================

EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-5.2-mini"

DOCS_DIR = "docs"
INDEX_FILE = "faiss.index"
META_FILE = "faiss_meta.pkl"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
BATCH_SIZE = 64
TOP_K = 4

client = OpenAI()

# =========================
# Chunking
# =========================

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

# =========================
# Build FAISS Index
# =========================

def build_index():
    docs = []
    metas = []

    paths = glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    if not paths:
        raise ValueError("No .txt files found in docs/ directory.")

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        for i, chunk in enumerate(chunk_text(text)):
            docs.append(chunk)
            metas.append({
                "source": os.path.basename(path),
                "chunk": i
            })

    # Generate embeddings in batches
    vectors = []
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch
        )
        vectors.extend([r.embedding for r in resp.data])

    if not vectors:
        raise ValueError("No embeddings were created.")

    vectors = np.array(vectors, dtype="float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # Persist index and metadata
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump({"docs": docs, "metas": metas}, f)

    print(f"✅ Built FAISS index with {len(docs)} chunks")
    return index, docs, metas

# =========================
# Load or Build Index
# =========================

def load_index():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        return build_index()

    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)

    return index, meta["docs"], meta["metas"]

index, docs, metas = load_index()

# =========================
# Retrieval
# =========================

def retrieve(query, k=TOP_K):
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=query
    )
    q_emb = np.array([resp.data[0].embedding], dtype="float32")
    faiss.normalize_L2(q_emb)

    scores, indices = index.search(q_emb, k)

    results = []
    for idx in indices[0]:
        results.append({
            "text": docs[idx],
            "meta": metas[idx]
        })

    return results

# =========================
# Prompt Construction
# =========================

def create_prompt(query, retrieved):
    context = "\n\n---\n\n".join(
        f"Source: {r['meta']['source']} (chunk {r['meta']['chunk']})\n{r['text']}"
        for r in retrieved
    )

    return f"""
You are a knowledgeable assistant.
Answer ONLY using the provided context.
If the answer is not present, say "I don't know."

Context:
{context}

Question:
{query}

Answer:
""".strip()

# =========================
# Answer Generation
# =========================

def answer(query):
    retrieved = retrieve(query)
    prompt = create_prompt(query, retrieved)

    resp = client.responses.create(
        model=LLM_MODEL,
        input=prompt,
        max_output_tokens=600
    )

    # Extract text from Responses API
    output_text = ""
    for item in resp.output:
        for block in item.get("content", []):
            if block.get("type") == "output_text":
                output_text += block.get("text", "")

    return output_text.strip()

# =========================
# Interactive CLI
# =========================

if __name__ == "__main__":
    print("📚 RAG system ready. Ask questions about your documents.")
    print("Type 'exit' to quit.")

    while True:
        q = input("\n> ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        print("\nAnswer:\n")
        print(answer(q))

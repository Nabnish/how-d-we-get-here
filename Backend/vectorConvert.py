import faiss
import numpy as np
import pickle
import hashlib
import time

# =====================================================
# CONFIG
# =====================================================

INDEX_FILE = "pubmed_faiss.index"
META_FILE = "pubmed_meta.pkl"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
BATCH_SIZE = 16
EMBED_DIM = 1536

# =====================================================
# LLM: Local Mistral 7B (ctransformers)
# =====================================================

from ctransformers import AutoModelForCausalLM

# Download/open GGUF file from HuggingFace:
# https://huggingface.co/TheBloke/openinstruct-mistral-7B-GGUF
llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/openinstruct-mistral-7B-GGUF",
    model_file="openinstruct-mistral-7b.Q4_K_M.gguf",
    model_type="mistral"
)

def local_mistral_answer(prompt: str) -> str:
    return llm(prompt)

# =====================================================
# STUB PUBMED CLIENT
# =====================================================

class PubMedAPI:
    def __init__(self, email=None):
        self.email = email

    def search_and_fetch(self, query, max_results=20):
        # Replace with real API calls if needed
        return [
            {
                "pmid": "0000001",
                "title": "Stub article for testing",
                "journal": "Test Journal",
                "full_text": "This is a short stub text used to test indexing and retrieval."
            }
        ]

# =====================================================
# DETERMINISTIC FALLBACK EMBEDDINGS
# =====================================================

def deterministic_embedding(text, dim=EMBED_DIM):
    h = hashlib.sha256(text.encode("utf-8")).digest()
    arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
    reps = int(np.ceil(dim / arr.size))
    vec = np.tile(arr, reps)[:dim]
    norm = np.linalg.norm(vec)
    if norm == 0:
        vec += 1e-6
        norm = np.linalg.norm(vec)
    return (vec / norm).astype("float32")

def get_embeddings(inputs):
    # Fallback: deterministic embeddings for simplicity
    vecs = [deterministic_embedding(t) for t in inputs]
    return np.array(vecs, dtype="float32")

# =====================================================
# CHUNKING
# =====================================================

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# =====================================================
# BUILD FAISS INDEX
# =====================================================

def build_pubmed_index(articles):
    texts = []
    metas = []

    for article in articles:
        for i, chunk in enumerate(chunk_text(article["full_text"])):
            texts.append(chunk)
            metas.append({
                "pmid": article["pmid"],
                "title": article["title"],
                "journal": article["journal"],
                "chunk": i
            })

    if not texts:
        raise ValueError("No text to embed.")

    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_vecs = get_embeddings(batch)
        vectors.extend(batch_vecs)

    vectors = np.array(vectors, dtype="float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump({"texts": texts, "metas": metas}, f)

    print(f"✅ Indexed {len(texts)} chunks")

# =====================================================
# SEARCH
# =====================================================

def search_pubmed(query, k=5):
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        data = pickle.load(f)

    q_emb = get_embeddings([query])
    faiss.normalize_L2(q_emb)
    _, indices = index.search(q_emb, k)

    return [
        {
            "text": data["texts"][i],
            "meta": data["metas"][i]
        }
        for i in indices[0]
    ]

# =====================================================
# RAG ANSWER
# =====================================================

def answer_with_pubmed(query):
    retrieved = search_pubmed(query)
    context = "\n\n---\n\n".join(
        f"{r['meta']['title']} (PMID: {r['meta']['pmid']})\n{r['text']}"
        for r in retrieved
    )

    prompt = f"""
You are a medical research assistant.
Answer ONLY using the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{query}

Answer:
""".strip()

    return local_mistral_answer(prompt)

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    pubmed = PubMedAPI(email="hrithikmadhu2008@gmail.com")

    articles = pubmed.search_and_fetch(
        query="diabetes treatment",
        max_results=10
    )

    build_pubmed_index(articles)

    print("\nAnswer:")
    print(answer_with_pubmed("What is this article about?"))

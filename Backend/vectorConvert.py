import faiss
import numpy as np
import pickle
import hashlib
import time
import os
import importlib.util
import sys

# Ensure console uses UTF-8 where supported (prevents UnicodeEncodeError on Windows)
try:
    # Python 3.7+: reconfigure stdout encoding to utf-8
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    # fallback: set PYTHONUTF8 env or run with -X utf8 if reconfigure not available
    os.environ.setdefault("PYTHONUTF8", "1")

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
# PUBMED CLIENT (use pdfreader.py output)
# =====================================================
# Replace the local stub with the real PubMedAPI from Backend/pdfreader.py.
try:
    from Backend.pdfreader import PubMedAPI
except Exception:
    # Fallback: load pdfreader.py by path (works even if Backend isn't a package)
    pdf_path = os.path.join(os.path.dirname(__file__), "pdfreader.py")
    spec = importlib.util.spec_from_file_location("pdfreader", pdf_path)
    pdfreader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pdfreader)
    PubMedAPI = getattr(pdfreader, "PubMedAPI")

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

    # Use plain ASCII to avoid encoding issues on Windows consoles
    print(f"Indexed {len(texts)} chunks")

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

# --- New helpers to keep prompt under the model's context window ---
# Rough token estimate: average characters per token ~= 4 (approx).
# Use a safety multiplier to avoid exceeding the model window.
TOKEN_CHAR_RATIO = 4.0
SAFETY_MULTIPLIER = 1.15

def estimate_tokens_by_chars(text: str) -> int:
    return max(1, int((len(text) / TOKEN_CHAR_RATIO) * SAFETY_MULTIPLIER))

# Model window and reserved space
MAX_MODEL_CONTEXT_TOKENS = 512
RESERVED_TOKENS_FOR_QUESTION_AND_ANSWER = 64
# Be conservative on chunk length
MAX_CHARS_PER_CHUNK = 180

def build_context_for_query(retrieved_chunks, question: str):
    # tokens available for context after reserving space for QA
    available_for_context = MAX_MODEL_CONTEXT_TOKENS - RESERVED_TOKENS_FOR_QUESTION_AND_ANSWER
    q_tokens = estimate_tokens_by_chars(question)
    overhead_margin = 16
    remaining = max(0, available_for_context - q_tokens - overhead_margin)

    # prepare parts (title + truncated text)
    parts = []
    for r in retrieved_chunks:
        title_meta = f"{r['meta'].get('title','')} (PMID: {r['meta'].get('pmid','')})\n"
        text = r["text"][:MAX_CHARS_PER_CHUNK].strip()
        chunk_text = title_meta + text
        toks = estimate_tokens_by_chars(chunk_text)
        parts.append({"text": chunk_text, "tokens": toks})

    # Greedy include until remaining, but allow trimming if still over budget
    included = []
    used = 0
    for p in parts:
        if used + p["tokens"] > remaining:
            break
        included.append(p["text"])
        used += p["tokens"]

    # If nothing fits, include a very small single chunk
    if not included and parts:
        single = parts[0]["text"][: int(MAX_CHARS_PER_CHUNK / 2)]
        included = [single]
        used = estimate_tokens_by_chars(single)

    # Final enforcement: if estimate still exceeds available, iteratively trim
    def current_context_text(lst):
        return "\n\n---\n\n".join(lst)

    # If still too large, drop last chunks until fits
    context_list = included.copy()
    est = estimate_tokens_by_chars(current_context_text(context_list))
    while est > remaining and len(context_list) > 1:
        context_list.pop()  # drop least relevant (last)
        est = estimate_tokens_by_chars(current_context_text(context_list))

    # If only one chunk and still too big, truncate it further
    if est > remaining and context_list:
        max_chars = max(50, int((remaining / estimate_tokens_by_chars(context_list[0])) * len(context_list[0]) * 0.9))
        context_list[0] = context_list[0][:max_chars]
        est = estimate_tokens_by_chars(current_context_text(context_list))

    context = current_context_text(context_list)
    return context

# Replace answer_with_pubmed with token-safe prompt builder
def answer_with_pubmed(query):
    retrieved = search_pubmed(query, k=10)  # retrieve up to 10 but we will limit by token budget
    context = build_context_for_query(retrieved, question=query)

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

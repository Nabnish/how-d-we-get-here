import faiss
import numpy as np
import pickle
import hashlib
import time
import os
import importlib.util
import sys
from pdfimport import PubMedAPI


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

BASE_DIR = os.path.dirname(__file__)
INDEX_FILE = os.path.join(BASE_DIR, "pubmed_faiss.index")
META_FILE = os.path.join(BASE_DIR, "pubmed_meta.pkl")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
BATCH_SIZE = 64  # Increased for better GPU utilization
EMBED_DIM = 384  # Reduced from 1536 (sentence-transformers default)

# =====================================================
# LLM: Local Mistral 7B (ctransformers) - robust imports
# =====================================================

_llm_load_error = None
_embed_load_error = None
llm = None
embedding_model = None

try:
    from ctransformers import AutoModelForCausalLM
    llm = AutoModelForCausalLM.from_pretrained(
        "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        model_file="mistral-7b-instruct-v0.1.Q3_K_S.gguf",
        model_type="mistral",
        gpu_layers=0  # CPU-only mode (set to >0 only if CUDA is installed)
    )
except Exception as e:
    _llm_load_error = str(e)

try:
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    _embed_load_error = str(e)


def local_mistral_answer(prompt: str) -> str:
    if llm is None:
        raise RuntimeError(f"LLM not available: {_llm_load_error or 'missing ctransformers or model file'}")
    return llm(prompt, max_new_tokens=256, temperature=0.7)  # Limit tokens for faster responses


# No PubMed dependency needed for file-based RAG

# =====================================================
# DETERMINISTIC FALLBACK EMBEDDINGS
# =====================================================

def get_embeddings(inputs):
    """Get embeddings for a list of texts.

    Preferred: use the SentenceTransformer model when available.
    Fallback: deterministic, hashing-based vectors (lower quality) so the app can run without heavy packages.
    """
    if embedding_model is not None:
        vecs = embedding_model.encode(inputs, show_progress_bar=False, batch_size=64)
        return np.array(vecs, dtype="float32")

    # Fallback deterministic embeddings (stable across runs)
    import warnings
    warnings.warn("Using fallback hashing embeddings. Install 'sentence-transformers' for better quality.")

    def text_to_vector(s: str, dim=EMBED_DIM):
        h = hashlib.sha256(s.encode("utf-8")).digest()
        # Expand hash to required dim with repeated hashing
        out = np.zeros(dim, dtype="float32")
        i = 0
        counter = 0
        while i < dim:
            chunk = hashlib.sha256(h + counter.to_bytes(2, "little")).digest()
            for b in chunk:
                if i >= dim:
                    break
                out[i] = (b - 128) / 128.0  # map byte to approx [-1,1]
                i += 1
            counter += 1
        # normalize
        norm = np.linalg.norm(out)
        if norm > 0:
            out /= norm
        return out

    vecs = [text_to_vector(s) for s in inputs]
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

    # Use approximate nearest neighbor search (IVFFlat) for faster retrieval
    # Falls back to IndexFlatIP for small indices
    if len(vectors) > 1000:
        nlist = max(10, len(vectors) // 100)  # Number of clusters
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(vectors.shape[1]), vectors.shape[1], nlist)
        index.train(vectors)
    else:
        index = faiss.IndexFlatIP(vectors.shape[1])
    
    index.add(vectors)

    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump({"texts": texts, "metas": metas}, f)

    # Use plain ASCII to avoid encoding issues on Windows consoles
    print(f"Indexed {len(texts)} chunks")
    return len(texts)

# =====================================================
# SEARCH
# =====================================================

def search_pubmed(query, k=5):
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        raise FileNotFoundError("Index files not found. Please upload a PDF first using /api/upload-pdf")

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
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        raise FileNotFoundError("No documents indexed. Please upload a PDF first using /api/upload-pdf")

    retrieved = search_pubmed(query, k=10)  # retrieve up to 10 but we will limit by token budget
    context = build_context_for_query(retrieved, question=query)

    prompt = f"""
You are a medical report explanation assistant.

You MUST answer strictly from the provided text.
If information is missing, say "Not mentioned in the document".
Do NOT add external medical knowledge.


STRICT RULES (must follow):
- Use ONLY the information present in the context.
- DO NOT assume missing tests.
- DO NOT invent lab values or test names.
- If the context does NOT contain laboratory test results, say EXACTLY:
  "This report does not contain laboratory test results."

Context:
{context}

Question:
{query}

Answer:
""".strip()

    try:
        return local_mistral_answer(prompt)
    except Exception as e:
        # Provide clearer runtime error so Flask can return 500 with helpful message
        raise RuntimeError(f"LLM inference failed: {type(e).__name__}: {e}")


def build_index_from_text(documents: list):
    """Build FAISS index from a list of document dictionaries.
    
    Args:
        documents: List of dicts with 'text' key (and optionally 'title', 'source')
    
    Returns:
        Dict with indexing results
    """
    if not documents:
        raise ValueError("No documents provided")
    
    texts = []
    metas = []
    
    for doc_idx, doc in enumerate(documents):
        text = doc.get("text", "")
        if not text:
            continue
        
        for chunk_idx, chunk in enumerate(chunk_text(text)):
            texts.append(chunk)
            metas.append({
                "doc_id": doc_idx,
                "title": doc.get("title", "Document"),
                "source": doc.get("source", "Uploaded"),
                "chunk": chunk_idx
            })
    
    if not texts:
        raise ValueError("No text content to index")
    
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_vecs = get_embeddings(batch)
        vectors.extend(batch_vecs)
    
    vectors = np.array(vectors, dtype="float32")
    faiss.normalize_L2(vectors)
    
    if len(vectors) > 1000:
        nlist = max(10, len(vectors) // 100)
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(vectors.shape[1]), vectors.shape[1], nlist)
        index.train(vectors)
    else:
        index = faiss.IndexFlatIP(vectors.shape[1])
    
    index.add(vectors)
    
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump({"texts": texts, "metas": metas}, f)
    
    print(f"Indexed {len(texts)} chunks from {len(documents)} documents")
    return {
        "indexed_chunks": len(texts),
        "index_file": INDEX_FILE,
        "meta_file": META_FILE,
        "documents_indexed": len(documents)
    }

# =====================================================
# MAIN
# =====================================================



    articles = pubmed.search_and_fetch(
        query="diabetes treatment",
        max_results=10
    )

    build_pubmed_index(articles)

    print("\nAnswer:")
    print(answer_with_pubmed("What is this article about?"))

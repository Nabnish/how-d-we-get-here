import faiss
import numpy as np
import pickle
import hashlib
import os
import sys
<<<<<<< HEAD
from pdfimport import PubMedAPI

=======
import warnings
>>>>>>> 1234217331549fca723921f3108659fe08491817

# =====================================================
# UTF-8 SAFE CONSOLE (Windows)
# =====================================================
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    os.environ.setdefault("PYTHONUTF8", "1")

# =====================================================
# CONFIG
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "rag_faiss.index")
META_FILE = os.path.join(BASE_DIR, "rag_meta.pkl")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
BATCH_SIZE = 64
EMBED_DIM = 384

# =====================================================
# OPTIONAL LOCAL LLM (CPU-SAFE)
# =====================================================

llm = None
embedding_model = None
_llm_load_error = None
_embed_load_error = None

try:
    from ctransformers import AutoModelForCausalLM
    llm = AutoModelForCausalLM.from_pretrained(
        "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        model_file="mistral-7b-instruct-v0.1.Q3_K_S.gguf",
        model_type="mistral",
        gpu_layers=0
    )
except Exception as e:
    _llm_load_error = str(e)

try:
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    _embed_load_error = str(e)


def local_mistral_answer(prompt: str) -> str:
    if llm is None:
        raise RuntimeError(
            f"Local LLM unavailable: {_llm_load_error or 'ctransformers not installed'}"
        )
    return llm(prompt, max_new_tokens=256, temperature=0.7)

# =====================================================
# EMBEDDINGS (ROBUST FALLBACK)
# =====================================================

def get_embeddings(texts):
    if embedding_model is not None:
        vecs = embedding_model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False
        )
        return np.asarray(vecs, dtype="float32")

    warnings.warn(
        "SentenceTransformer not installed. "
        "Using low-quality hashing embeddings."
    )

    def hash_embed(text, dim=EMBED_DIM):
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = np.zeros(dim, dtype="float32")
        i = 0
        ctr = 0
        while i < dim:
            chunk = hashlib.sha256(h + ctr.to_bytes(2, "little")).digest()
            for b in chunk:
                if i >= dim:
                    break
                vec[i] = (b - 128) / 128.0
                i += 1
            ctr += 1
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    return np.array([hash_embed(t) for t in texts], dtype="float32")

# =====================================================
# TEXT CHUNKING
# =====================================================

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + size, length)
        chunks.append(text[start:end])
        start += size - overlap

    return chunks

# =====================================================
# BUILD INDEX FROM GENERIC TEXT (USED BY FLASK)
# =====================================================

def build_index_from_text(documents):
    if not documents:
        raise ValueError("No documents provided")

    texts, metas = [], []

    for doc_id, doc in enumerate(documents):
        raw_text = doc.get("text", "")
        if not raw_text.strip():
            continue

        for idx, chunk in enumerate(chunk_text(raw_text)):
            texts.append(chunk)
            metas.append({
                "doc_id": doc_id,
                "title": doc.get("title", "Document"),
                "source": doc.get("source", "Uploaded"),
                "chunk": idx
            })

    if not texts:
        raise ValueError("No text chunks created")

    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        vectors.extend(get_embeddings(texts[i:i + BATCH_SIZE]))

    vectors = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(vectors)

    if len(vectors) > 1000:
        nlist = max(10, len(vectors) // 100)
        index = faiss.IndexIVFFlat(
            faiss.IndexFlatIP(vectors.shape[1]),
            vectors.shape[1],
            nlist
        )
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
        "documents_indexed": len(documents),
        "index_file": INDEX_FILE,
        "meta_file": META_FILE
    }

# =====================================================
# SEARCH
# =====================================================

def search_index(query, k=5):
    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError("FAISS index not found")

    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        data = pickle.load(f)

    q_emb = get_embeddings([query])
    faiss.normalize_L2(q_emb)

    _, idxs = index.search(q_emb, k)

    return [
        {
            "text": data["texts"][i],
            "meta": data["metas"][i]
        }
        for i in idxs[0]
    ]

# =====================================================
# RAG ANSWERING
# =====================================================

def answer_with_pubmed(question: str):
    retrieved = search_index(question, k=8)

    if not retrieved:
        return "I don't know."

    context = "\n\n".join(
        r["text"][:300] for r in retrieved
    )

    prompt = f"""
<<<<<<< HEAD
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
=======
You are a medical assistant.
Answer ONLY using the context below.
If the answer is not present, say "I don't know."
>>>>>>> 1234217331549fca723921f3108659fe08491817

Context:
{context}

Question:
{question}

Answer:
""".strip()

    return local_mistral_answer(prompt)

# =====================================================
# EXTRACTIVE FALLBACK
# =====================================================

<<<<<<< HEAD

=======
def extractive_answer(query: str, k=5):
    retrieved = search_index(query, k)
>>>>>>> 1234217331549fca723921f3108659fe08491817

    if not retrieved:
        return "No relevant information found."

    parts = []
    for r in retrieved:
        meta = r["meta"]
        parts.append(
            f"{meta.get('title','')}\n{r['text']}"
        )

    return "\n\n---\n\n".join(parts)

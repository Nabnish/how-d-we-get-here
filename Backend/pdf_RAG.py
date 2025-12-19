import os
import faiss
import pickle
import numpy as np
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

# =============================
# CONFIG
# =============================
UPLOAD_DIR = "uploads"
INDEX_FILE = "pdf.index"
META_FILE = "pdf_meta.pkl"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBED_DIM = 384

os.makedirs(UPLOAD_DIR, exist_ok=True)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# =============================
# PDF EXTRACTION
# =============================

def extract_text_pdf(path: str) -> str:
    text = ""

    # Try normal text extraction
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    # OCR fallback if text is too small
    if len(text.strip()) < 1000:
        images = convert_from_path(path)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img)

    return clean_text(text)


def clean_text(text: str) -> str:
    lines = text.splitlines()
    lines = [l.strip() for l in lines if len(l.strip()) > 5]
    lines = [l for l in lines if not l.lower().startswith("page")]
    return "\n".join(lines)

# =============================
# CHUNKING (Sentence-aware)
# =============================

def chunk_text(text: str):
    sentences = sent_tokenize(text)
    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) <= CHUNK_SIZE:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s

    if current:
        chunks.append(current.strip())

    return chunks

# =============================
# BUILD FAISS INDEX
# =============================

def build_index_from_pdf(pdf_path: str):
    text = extract_text_pdf(pdf_path)
    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("No extractable text found")

    vectors = embedding_model.encode(chunks, convert_to_numpy=True)
    vectors = vectors.astype("float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)

    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(chunks, f)

    return len(chunks)

# =============================
# SEARCH
# =============================

def search(query: str, k=5):
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        chunks = pickle.load(f)

    q_vec = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)

    _, idx = index.search(q_vec, k)

    return [chunks[i] for i in idx[0]]

# =============================
# RAG ANSWER
# =============================

from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    model_file="mistral-7b-instruct-v0.1.Q3_K_S.gguf",
    model_type="mistral"
)

def answer(query: str):
    retrieved = search(query, k=6)

    context = "\n\n---\n\n".join(retrieved[:4])

    prompt = f"""
You are a medical research assistant.
Answer ONLY from the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
""".strip()

    return llm(prompt, max_new_tokens=256, temperature=0.3)

# =============================
# TEST
# =============================

if __name__ == "__main__":
    pdf_path = "uploads/sample.pdf"
    print("Indexing PDF...")
    build_index_from_pdf(pdf_path)

    while True:
        q = input("\nAsk a question (or 'exit'): ")
        if q.lower() == "exit":
            break
        print(answer(q))

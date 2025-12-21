from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Upload configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

# Ensure upload directory exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


def allowed_file(filename: str) -> bool:
    return (
        isinstance(filename, str)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ===============================
# LLM Configuration
# ===============================
LLM_METHOD = os.getenv("LLM_METHOD", "ollama")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"


# ===============================
# LLM Functions
# ===============================
def classify_document(text: str) -> str:
    text = text.lower()

    lab_keywords = [
        "hemoglobin", "wbc", "rbc", "platelet",
        "cbc", "mg/dl", "g/dl", "cells/mcl"
    ]

    mental_keywords = [
        "mental capacity", "dementia", "stroke",
        "cognitive", "orientation", "incontinent"
    ]

    lab_score = sum(1 for k in lab_keywords if k in text)
    mental_score = sum(1 for k in mental_keywords if k in text)

    if lab_score >= 2:
        return "LAB_REPORT"
    if mental_score >= 2:
        return "MENTAL_CAPACITY_REPORT"

    return "UNKNOWN"

def chat_with_ollama(user_message: str, system_message: str) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}")


def chat_with_mistral_api(user_message: str, system_message: str) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}"
    }

    payload = {
        "model": "mistral-7b-instruct",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }

    response = requests.post(
        MISTRAL_API_URL,
        json=payload,
        headers=headers,
        timeout=120
    )
    response.raise_for_status()
    data = response.json()

    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def is_ollama_available(timeout: int = 5) -> (bool, str):
    """Check if Ollama service is reachable and responsive."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout)
        resp.raise_for_status()
        return True, "Ollama reachable"
    except requests.RequestException as e:
        return False, str(e)


@app.route("/api/llm/status", methods=["GET"])
def llm_status():
    """Return LLM readiness info (useful for frontend/health checks)."""
    if LLM_METHOD == "ollama":
        ok, info = is_ollama_available()
        return jsonify({"method": "ollama", "ready": ok, "details": info})
    elif LLM_METHOD == "mistral_api":
        return jsonify({"method": "mistral_api", "ready": bool(MISTRAL_API_KEY)})
    else:
        return jsonify({"method": LLM_METHOD, "ready": False}), 400


# ===============================
# API Routes
# ===============================
@app.route("/api/chat", methods=["POST"])
def chat():
    from vectorConvert import search_pubmed, build_context_for_query
    
    data = request.get_json(force=True)
    user_message = data.get("message", "").strip()
    ignore_rag = bool(data.get("ignore_rag", False))

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    system_message = (
        "You explain medical information clearly and safely to patients. "
        "Always encourage consulting a healthcare professional. "
        "Only use the provided context. If context is missing, say so."
    )

    # Use RAG to retrieve relevant chunks from indexed PDF
    full_user_message = user_message
    if not ignore_rag:
        try:
            retrieved_chunks = search_pubmed(user_message, k=5)
            if retrieved_chunks:
                context = build_context_for_query(retrieved_chunks, user_message)
                full_user_message = (
                    f"Using this medical document context:\n\n{context}\n\n"
                    f"Question: {user_message}"
                )
        except Exception as e:
            print(f"RAG retrieval warning: {e}")
            # Continue with just the user message if RAG fails

    try:
        if LLM_METHOD == "ollama":
            reply = chat_with_ollama(full_user_message, system_message)
        elif LLM_METHOD == "mistral_api":
            reply = chat_with_mistral_api(full_user_message, system_message)
        else:
            return jsonify({"error": "Invalid LLM_METHOD"}), 500

        return jsonify({"response": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload-pdf", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        import fitz  # PyMuPDF
        from cleanup import clean_medical_text
        from nlp import extract_lab_results
        from vectorConvert import build_index_from_text

        # 1️⃣ Extract raw text from PDF
        raw_text = ""
        with fitz.open(file_path) as pdf:
            for page in pdf:
                raw_text += page.get_text()

        if not raw_text.strip():
            return jsonify({"error": "PDF contains no extractable text"}), 400

        # 2️⃣ Clean text
        clean_text = clean_medical_text(raw_text)

        if len(clean_text) < 300:
            return jsonify({"error": "PDF text too short or unreadable"}), 400

        # 3️⃣ Extract NLP results (structured data)
        nlp_output = extract_lab_results(clean_text)

        # 4️⃣ Document classification
        doc_type = classify_document(clean_text)

        # 5️⃣ Format structured NLP data
        structured_text = ""
        if nlp_output.get("lab_results"):
            structured_lines = ["=== EXTRACTED LAB RESULTS ==="]
            for lab in nlp_output["lab_results"]:
                structured_lines.append(
                    f"Test: {lab['test_name']}, "
                    f"Value: {lab['value']} {lab['unit']}, "
                    f"Reference: {lab['reference_range']}, "
                    f"Status: {lab['status']}"
                )
            structured_text = "\n".join(structured_lines)

        # 6️⃣ BUILD INDEX: Send structured + cleaned text to RAG (FAISS)
        document_for_rag = {
            "text": f"{structured_text}\n\n{clean_text}",
            "title": filename,
            "source": "Medical PDF Upload",
            "doc_type": doc_type
        }

        index_result = build_index_from_text([document_for_rag])

        return jsonify({
            "status": "success",
            "filename": filename,
            "document_type": doc_type,
            "extracted_chars": len(clean_text),
            "labs_detected": len(nlp_output.get("lab_results", [])),
            "nlp_output": nlp_output,
            "rag_indexed": index_result,
            "message": "PDF processed: NLP extracted → indexed to RAG → ready for chat queries"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    llm_info = {"method": LLM_METHOD}
    if LLM_METHOD == "ollama":
        ok, info = is_ollama_available()
        llm_info.update({"ready": ok, "details": info})
    elif LLM_METHOD == "mistral_api":
        llm_info.update({"ready": bool(MISTRAL_API_KEY), "api_key_set": bool(MISTRAL_API_KEY)})

    return jsonify({
        "status": "ok",
        "llm": llm_info
    })


# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

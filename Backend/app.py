from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

import pdfplumber
import pytesseract
from pdf2image import convert_from_path

# ===============================
# ENV + APP INIT
# ===============================
load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===============================
# HELPERS
# ===============================

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_pdf_text(path: str) -> str:
    """
    Extract text from PDF.
    Uses pdfplumber first, falls back to OCR if needed.
    """
    text = ""

    # 1️⃣ Try text layer
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    # 2️⃣ OCR fallback for scanned PDFs
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


# ===============================
# LLM CONFIG
# ===============================
LLM_METHOD = os.getenv("LLM_METHOD", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"


def chat_with_ollama(user_message: str, system_message: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "stream": False
    }

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=120
    )
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "")


def is_ollama_available(timeout=5):
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout)
        r.raise_for_status()
        return True, "Ollama running"
    except Exception as e:
        return False, str(e)


# ===============================
# ROUTES
# ===============================

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    msg = data.get("message", "").strip()

    if not msg:
        return jsonify({"error": "Message required"}), 400

    system = (
        "You are a medical assistant. "
        "Answer clearly and safely. "
        "Always advise consulting a healthcare professional."
    )

    try:
        reply = chat_with_ollama(msg, system)
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
        return jsonify({"error": "Only PDF files allowed"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    
    # Extract text (fast)
    import pdfplumber
    extracted_text = ""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            extracted_text += page.extract_text() or ""

    if not extracted_text.strip():
        return jsonify({"error": "No text extracted"}), 400

    # 🔥 Run indexing in background
    from threading import Thread
    from vectorConvert import build_index_from_text

    def index_job():
        try:
            build_index_from_text([{
                "text": extracted_text,
                "title": filename,
                "source": "Uploaded PDF"
            }])
            print("Indexing completed")
        except Exception as e:
            print("Indexing failed:", e)

    Thread(target=index_job, daemon=True).start()

    return jsonify({
        "status": "processing",
        "message": "PDF uploaded. Indexing in progress.",
        "filename": filename
    }), 202



@app.route("/api/query", methods=["POST"])
def query_docs():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question required"}), 400

    try:
        from vectorConvert import answer_with_pubmed, extractive_answer
        use_llm = bool(data.get("use_llm", True))

        answer = (
            answer_with_pubmed(question)
            if use_llm else
            extractive_answer(question)
        )

        return jsonify({
            "question": question,
            "answer": answer
        })

    except FileNotFoundError:
        return jsonify({
            "error": "No PDF indexed. Upload a PDF first."
        }), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    ok, info = is_ollama_available()
    return jsonify({
        "status": "ok",
        "llm": {
            "method": LLM_METHOD,
            "ready": ok,
            "details": info
        }
    })


# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

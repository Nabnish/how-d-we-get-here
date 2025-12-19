from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import pdfplumber
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

    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    return data.get("message", {}).get("content", "")


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


# ===============================
# API Routes
# ===============================
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    system_message = (
        "You explain medical information clearly and safely to patients. "
        "Always encourage consulting a healthcare professional."
    )

    if LLM_METHOD == "ollama":
        reply = chat_with_ollama(user_message, system_message)
    elif LLM_METHOD == "mistral_api":
        reply = chat_with_mistral_api(user_message, system_message)
    else:
        return jsonify({"error": "Invalid LLM_METHOD"}), 500

    return jsonify({"response": reply})


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

    extracted_text = ""

    # 🔹 pdfplumber extraction
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            extracted_text += page.extract_text() or ""

    return jsonify({
        "filename": filename,front
        "text": extracted_text[:8000]  # safety limit
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "llm_method": LLM_METHOD
    })


# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

# Medical Document RAG System

Simple backend for uploading PDFs and getting AI-powered answers about their content.

## Quick Start

### 1. Setup Backend

```bash
# Navigate to Backend folder
cd Backend

# Install dependencies
pip install -r requirements.txt

# Run Flask server
python app.py
```

Server runs on `http://localhost:5000`

### 2. API Endpoints

#### **Upload PDF & Create Index**
```bash
POST /api/upload-pdf
Content-Type: multipart/form-data

Body:
- file: [your_document.pdf]

Response:
{
  "status": "success",
  "filename": "document.pdf",
  "indexed_chunks": 42,
  "message": "PDF indexed successfully with 42 chunks"
}
```

#### **Query the Indexed Document**
```bash
POST /api/query
Content-Type: application/json

Body:
{
  "question": "What are the main topics covered?"
}

Response:
{
  "question": "What are the main topics covered?",
  "answer": "The document covers..."
}
```

#### **Health Check**
```bash
GET /api/health

Response:
{
  "status": "ok",
  "llm_method": "ollama"
}
```

---

## Configuration

Create/edit `Backend/.env`:

```env
# Choose: ollama (local) or mistral_api (cloud)
LLM_METHOD=ollama

# Ollama settings (if using local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b

# Mistral API key (if using cloud)
MISTRAL_API_KEY=your_api_key_here
```

---

## How It Works

1. **Upload PDF** → Backend extracts text
2. **Create Index** → Text is split into chunks and embedded
3. **Store Index** → FAISS index + metadata saved locally
4. **Query** → User question is embedded, similar chunks retrieved, LLM generates answer

---

## Frontend Integration

Upload a PDF and ask questions:

```javascript
// 1. Upload PDF
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:5000/api/upload-pdf', {
  method: 'POST',
  body: formData
});

// 2. Ask question
fetch('http://localhost:5000/api/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question: 'What is this about?' })
});
```

---

## Requirements

- Python 3.8+
- 4GB RAM minimum
- GPU optional (uses CPU by default)

---

## Files

- `app.py` - Flask backend with endpoints
- `vectorConvert.py` - Embeddings & FAISS indexing
- `requirements.txt` - Python dependencies
- `.env` - Configuration


#!/bin/bash
# Quick Setup Guide for Medical Records RAG System

## Step 1: Navigate to Backend Directory
```bash
cd "C:\Users\Nabeel\Documents\pratice programs\Project\Medical_records\how-d-we-get-here\backend"
```

## Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

## Step 3: (Optional) Install Tesseract-OCR for Scanned PDFs

### Windows:
1. Download: https://github.com/UB-Mannheim/tesseract/wiki
2. Run installer (default location is fine)
3. Add to system PATH if needed

### Linux:
```bash
sudo apt-get install tesseract-ocr
```

### Mac:
```bash
brew install tesseract
```

## Step 4: Setup LLM - Choose One Option

### Option A: Local LLM (Ollama) - RECOMMENDED
```bash
# 1. Download Ollama from https://ollama.ai
# 2. Install and start Ollama service

# 3. Pull the Mistral model
ollama pull mistral:7b

# 4. Create .env file in backend folder:
echo LLM_METHOD=ollama > .env
echo OLLAMA_BASE_URL=http://localhost:11434 >> .env
echo OLLAMA_MODEL=mistral:7b >> .env
```

### Option B: Cloud LLM (Mistral API)
```bash
# 1. Get API key from https://console.mistral.ai

# 2. Create .env file in backend folder:
echo LLM_METHOD=mistral_api > .env
echo MISTRAL_API_KEY=your_api_key_here >> .env
```

## Step 5: Test the Installation
```bash
# Test basic vector conversion
python test_vectorconvert.py

# Test end-to-end with sample PDF
python test_upload_and_query.py
```

## Step 6: Run the Backend Server
```bash
python app.py
```

Server will be running at: http://localhost:5000

## Step 7: Test Health Endpoint
```bash
# Open in browser or terminal:
curl http://localhost:5000/api/health
```

Expected response:
```json
{
  "status": "ok",
  "llm": {
    "method": "ollama",
    "ready": true,
    "details": "Ollama running"
  }
}
```

## Troubleshooting

### Import Error: "No module named 'X'"
Solution: Reinstall dependencies
```bash
pip install --upgrade -r requirements.txt
```

### OCR Not Working
- Verify pytesseract is installed: `pip install pytesseract`
- Install Tesseract-OCR (see Step 3)
- System will fall back to text-only extraction if unavailable

### Ollama Connection Failed
- Make sure Ollama service is running
- Check: `ollama list` shows your models
- Verify OLLAMA_BASE_URL in .env matches your setup

### Mistral API Errors
- Verify API key is correct in .env
- Check API quota at https://console.mistral.ai
- Ensure internet connection is active

### PDF Processing Errors
- Ensure PDF is valid and not corrupted
- Try a different PDF file
- Check available disk space in uploads/ folder

## System Requirements

- Python 3.8+
- 4GB RAM (8GB recommended for large PDFs)
- ~2GB disk space for dependencies
- GPU optional (system uses CPU by default)

## Key Files

- `app.py` - Flask backend with API endpoints
- `vectorConvert.py` - FAISS indexing and RAG
- `cleanup.py` - Text preprocessing
- `nlp.py` - Lab result extraction
- `pdfimport.py` - PubMed API integration
- `.env` - Configuration (create this file)
- `uploads/` - Folder for uploaded PDFs

## API Endpoints

```
POST /api/chat - Chat with RAG support
POST /api/upload-pdf - Upload and index PDF
POST /api/query - Query indexed documents
GET /api/health - Health check
```

See README.md for detailed API documentation.

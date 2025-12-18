# Mistral 7B Setup Guide

This backend now supports Mistral 7B through two methods:

## Option 1: Ollama (Local - Recommended)

### Step 1: Install Ollama
- **Windows/Mac/Linux**: Download from https://ollama.ai
- Install and start the Ollama service

### Step 2: Pull Mistral 7B Model
```bash
ollama pull mistral:7b
```

### Step 3: Configure Environment
Create a `.env` file in the `backend` folder:
```
LLM_METHOD=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b
```

### Step 4: Start Backend
```bash
python app.py
```

## Option 2: Mistral AI API (Cloud)

### Step 1: Get API Key
1. Go to https://console.mistral.ai
2. Sign up/login
3. Get your API key

### Step 2: Configure Environment
Create a `.env` file in the `backend` folder:
```
LLM_METHOD=mistral_api
MISTRAL_API_KEY=your_api_key_here
```

### Step 3: Start Backend
```bash
python app.py
```

## Testing

Test the setup by visiting:
```
http://localhost:5000/api/health
```

You should see:
```json
{
  "status": "ok",
  "llm_method": "ollama"  // or "mistral_api"
}
```

## Troubleshooting

### Ollama Issues:
- Make sure Ollama is running: `ollama list` should show your models
- Check if Ollama is accessible: `curl http://localhost:11434/api/tags`
- If using a different port, update `OLLAMA_BASE_URL` in `.env`

### Mistral API Issues:
- Verify your API key is correct
- Check your API quota/limits at https://console.mistral.ai
- Ensure you have internet connection

## Model Variants

For Ollama, you can use different Mistral variants:
- `mistral:7b` - Standard Mistral 7B
- `mistral:7b-instruct` - Instruction-tuned version
- `mistral:7b-instruct-q4_K_M` - Quantized version (smaller, faster)

Update `OLLAMA_MODEL` in `.env` to use a different variant.


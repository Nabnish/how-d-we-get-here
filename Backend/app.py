from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration - Choose your method: 'ollama' or 'mistral_api'
LLM_METHOD = os.getenv('LLM_METHOD', 'ollama')  # Default to Ollama

# Ollama configuration
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'mistral:7b')

# Mistral AI API configuration (alternative)
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY', '')
MISTRAL_API_URL = 'https://api.mistral.ai/v1/chat/completions'

def chat_with_ollama(user_message, system_message):
    url = "http://localhost:11434/api/chat"

    payload = {
        "model": "mistral:7b",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "stream": False
    }

    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    return data["message"]["content"]


def chat_with_mistral_api(user_message, system_message):
    """Use Mistral AI API (cloud-based)"""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MISTRAL_API_KEY}"
        }
        
        payload = {
            "model": "mistral-7b-instruct",  # or "mistral-tiny", "mistral-small", etc.
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        response = requests.post(MISTRAL_API_URL, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        return data.get('choices', [{}])[0].get('message', {}).get('content', 'No response generated')
    
    except Exception as e:
        print(f"Mistral API error: {str(e)}")
        raise

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({'error': 'Message required'}), 400

    system_message = "You explain medical information clearly to patients."

    reply = chat_with_ollama(user_message, system_message)
    return jsonify({"response": reply})

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'llm_method': LLM_METHOD})



if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')

from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # You can change this to gpt-4 or other models
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful medical research assistant. You provide accurate, evidence-based information about medical topics, treatments, and research. Always cite sources when possible and remind users to consult healthcare professionals for medical advice."
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        assistant_message = response.choices[0].message.content
        
        return jsonify({
            'response': assistant_message
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'An error occurred processing your request'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')


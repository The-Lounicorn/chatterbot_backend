from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

# Read Hugging Face API key from environment variable
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

conversation_history = []
MAX_TURNS = 6

@app.route('/', methods=['GET', 'HEAD'])
def home():
    return jsonify({"status": "Backend is running"}), 200

@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    try:
        data = request.get_json(force=True)
        input_text = data.get('prompt', '').strip()
        if not input_text:
            return jsonify({'error': 'Empty input'}), 400

        # Maintain conversation history with a max number of turns
        if len(conversation_history) > MAX_TURNS:
            conversation_history[:] = conversation_history[-MAX_TURNS:]

        formatted_history = ""
        for i, turn in enumerate(conversation_history):
            role = "User" if i % 2 == 0 else "Bot"
            formatted_history += f"{role}: {turn} "

        # Prepare payload for Hugging Face API
        payload = {"inputs": formatted_history + f"User: {input_text}"}
        response = requests.post(API_URL, headers=HEADERS, json=payload)

        try:
            response_json = response.json()
        except ValueError:
            # API returned something that's not JSON
            return jsonify({'error': 'Invalid response from Hugging Face API'}), 502

        # Extract response safely
        if isinstance(response_json, list) and len(response_json) > 0 and 'generated_text' in response_json[0]:
            bot_response = response_json[0]['generated_text']
        else:
            bot_response = "Sorry, I couldn't respond."

        conversation_history.append(input_text)
        conversation_history.append(bot_response)

        return jsonify({'response': bot_response})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

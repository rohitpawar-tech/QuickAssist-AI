import os
import logging
import json
import random
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Chatbot Logic Class ---
class ChatbotEngine:
    def __init__(self, intents_path):
        self.intents_path = intents_path
        self.intents = self.load_intents()
        self.vectorizer = None
        self.pattern_vectors = None
        self.patterns = []
        self.tags = []
        self.train_model()

    def load_intents(self):
        try:
            with open(self.intents_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            logger.error(f"Error loading intents: {e}")
            return {"intents": []}

    def train_model(self):
        """Pre-processes data and trains the TF-IDF vectorizer."""
        patterns = []
        tags = []
        
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                patterns.append(pattern)
                tags.append(intent['tag'])
        
        self.patterns = patterns
        self.tags = tags
        
        if not patterns:
            logger.warning("No patterns found in intents.json")
            return

        # Initialize and fit TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer()
        self.pattern_vectors = self.vectorizer.fit_transform(patterns)
        logger.info("Chatbot model trained successfully.")

    def get_response(self, user_input):
        """Finds the best response using Cosine Similarity."""
        if not self.vectorizer or not self.pattern_vectors.any():
            return "I'm sorry, the system is not fully loaded yet."

        # Log the query
        logger.info(f"User Query: {user_input}")

        # Transform user input
        user_vector = self.vectorizer.transform([user_input])
        
        # Calculate similarity
        similarities = cosine_similarity(user_vector, self.pattern_vectors)
        
        # Get index of highest similarity
        best_match_index = np.argmax(similarities)
        best_match_score = similarities[0][best_match_index]

        # Threshold for confidence (adjust as needed)
        THRESHOLD = 0.3
        
        if best_match_score > THRESHOLD:
            tag = self.tags[best_match_index]
            for intent in self.intents['intents']:
                if intent['tag'] == tag:
                    return random.choice(intent['responses'])
        
        # Fallback response
        return "I'm not sure I understand. Could you please rephrase that?"

# Initialize Chatbot
intents_file = os.path.join(os.path.dirname(__file__), 'intents.json')
chatbot = ChatbotEngine(intents_file)

# --- Routes ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Invalid input"}), 400
        
        user_message = data['message'].strip()
        
        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        bot_response = chatbot.get_response(user_message)
        
        return jsonify({"response": bot_response})

    except Exception as e:
        logger.error(f"Error in /chat route: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
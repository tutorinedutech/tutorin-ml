from flask import Flask, request, jsonify
import tensorflow as tf
import keras
import keras_nlp
import numpy as np
from src.model.siamese import SiameseModel

# Initialize Flask app
app = Flask(__name__)

# Load the Siamese model
siamese_model = SiameseModel()

@app.route('/encode', methods=['POST'])
def encode():
    data = request.get_json(force=True)
    sentence = data.get('sentence', None)
    if sentence is None:
        return jsonify({"error": "Please provide a sentence"}), 400

    sentence_tensor = tf.convert_to_tensor([[sentence]])  # Add batch and sentence dimension
    encoded_output = siamese_model.get_encoder()(sentence_tensor)
    return jsonify({"encoded_output": encoded_output.numpy().tolist()})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    sentences = data.get('sentences', None)
    if sentences is None or len(sentences) != 2:
        return jsonify({"error": "Please provide exactly 2 sentences"}), 400

    sentences_tensor = tf.convert_to_tensor([sentences])  # Add batch dimension
    similarity_score = siamese_model.model(sentences_tensor)
    return jsonify({"similarity_score": similarity_score.numpy().tolist()})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8601)
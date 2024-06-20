from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

model_path = ""
model = tf.saved_model.load()
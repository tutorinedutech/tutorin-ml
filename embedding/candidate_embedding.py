import tensorflow as tf
import pandas as pd
import json

# Assuming load_data function is correctly defined
from src.data.data_prep import load_data

# Load the TensorFlow model
model_path = './save_models/candidate_models/1'
loaded_model = tf.saved_model.load(model_path)

# Define the data path
data_path = 'E:\\Projects\\tutorin-ml\\data_source\\Data_RecomenderSystem.xlsx'

# Load data
similar_df, mentor_df = load_data(data_path=data_path)

# Define a function to handle NaNs and infinities if needed
def tf_if_null_return_zero(val):
    """
    This function replaces NaNs with zeros to clean the embedding inputs.
    """
    return tf.clip_by_value(val, -1e12, 1e12)

# Generate embeddings for mentor_ids in mentor_df
embeddings = []
mentor_ids = []
# Save embeddings to a JSON file
output_file = "mentor_embeddings.json"

with open(output_file, 'w') as f:
    for mentor_id in mentor_df['mentor_id'].values.tolist():
        embedding = loaded_model(tf.constant([f'{mentor_id}']))
        reshape_embedding = tf.reshape(embedding, (-1,)).numpy()
        f.write('{"id": "' + str(mentor_id) + '", "embedding":[' + ",".join(str(x) for x in list(reshape_embedding))+ ']}')
        f.write("\n")
print(f"Embeddings saved to {output_file}")


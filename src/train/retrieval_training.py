from src.data.data_prep import load_data, get_tensor_data, split_data
import numpy as np
import tensorflow as tf
from src.model.retrieval_model import RecomenderMentorModel
import tensorflow_recommenders as tfrs
import os

path = 'E:\\Projects\\tutorin-ml\\data_source\\Data_RecomenderSystem.xlsx'
#load data
similar_df, mentor_df = load_data(data_path=path)

similar_ds, mentor_ds = get_tensor_data(similar_df, mentor_df)

train, test = split_data(similar_ds, train_split=0.7, random_seed=42)

user_ids = similar_ds.batch(10).map(lambda x: x['user_id'])
mentor_ids = mentor_ds.batch(10)
kriteria_mentor_user = similar_ds.batch(10).map(lambda x: x['kriteria_mentor_user'])
#kriteria_mentor = mentor.batch(10).map(lambda x: x['kriteria_mentor'])

unique_user_ids = np.unique(np.concatenate(list(user_ids)))
unique_mentor_ids = np.unique(np.concatenate(list(mentor_ids)))
unique_kriteria_mentor_user = np.unique(np.concatenate(list(kriteria_mentor_user)))
#unique_kriteria_mentor = np.unique(np.concatenate(list(kriteria_mentor)))

model = RecomenderMentorModel(similar_ds, mentor_ds, unique_kriteria_mentor_user, unique_mentor_ids)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

model.fit(train, epochs=10)

train_accuracy = model.evaluate(
    train, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]
test_accuracy = model.evaluate(
    test, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]

print(f"Top-100 accuracy (train): {train_accuracy:.2f}.")
print(f"Top-100 accuracy (test): {test_accuracy:.2f}.")

# Create a model that takes in raw query features, and
index = tfrs.layers.factorized_top_k.BruteForce(model.query_model)
# recommends movies out of the entire movies dataset.
index.index_from_dataset(
  tf.data.Dataset.zip((mentor_ds.batch(100), mentor_ds.batch(100).map(model.candidate_model)))
)
# Get recommendations.
_, mentor_id = index({'kriteria_mentor_user': tf.constant(['Saya ingin belajar fisika dengan mahasiswa fisika yang sudah berpengalaman menjadi asisten dosen dan juga di daerah Yogyakarta dengan menggunakan bahasa indonesia harus laki-laki.'])})
print(f"mentor_id: {mentor_id[0][:3]}")

path = './save_models/retrieval_models/1'

#save query model
path_query = './save_models/query_models/1'
tf.saved_model.save(model.query_model, path_query)
#save candidates model
path_candidate = './save_models/candidate_models/1'
tf.saved_model.save(model.candidate_model, path_candidate)

# Save the index.
tf.saved_model.save(index, path)

# Load it back; can also be done in TensorFlow Serving.
loaded = tf.saved_model.load(path)

# Pass a user id in, get top predicted movie titles back.
scores, mentor_id = loaded({'kriteria_mentor_user': tf.constant(['Saya ingin belajar fisika dengan mahasiswa fisika yang sudah berpengalaman menjadi asisten dosen dan juga di daerah Yogyakarta dengan menggunakan bahasa indonesia harus laki-laki.'])})

print(f"Recommendations: {mentor_id[0][:3]}")
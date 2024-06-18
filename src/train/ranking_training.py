from src.data.data_prep import load_data, get_tensor_data, split_data
import numpy as np
import tensorflow as tf
from src.model.ranking_model import MentorModel
import tensorflow_recommenders as tfrs
import os

path = 'E:\\Projects\\tutorin-ml\\data_source\\Data_RecomenderSystem.xlsx'
#load data
similar_df, mentor_df = load_data(data_path=path)

similar_ds, mentor_ds = get_tensor_data(similar_df, mentor_df)

similar_ds = similar_ds.map(lambda x: {'kriteria_mentor_user': x['kriteria_mentor_user'],
                                       'kriteria_mentor': x['kriteria_mentor'],
                                       'similarity': x['similarity']})

train, test = split_data(similar_ds, train_split=0.7, random_seed=42)


kriteria_mentor_user = similar_ds.batch(10).map(lambda x: x['kriteria_mentor_user'])
kriteria_mentor = similar_ds.batch(10).map(lambda x: x['kriteria_mentor'])

unique_kriteria_mentor_user = np.unique(np.concatenate(list(kriteria_mentor_user)))
unique_kriteria_mentor = np.unique(np.concatenate(list(kriteria_mentor)))

model = MentorModel(similar_ds, 10_000)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

model.fit(train, epochs=10)

path = './save_models/ranking_models/1'
tf.saved_model.save(model, path)

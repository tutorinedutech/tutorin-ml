from src.data.data_prep import load_data, get_tensor_data, split_data
import numpy as np
import tensorflow as tf
from src.model.retrieval_model import RecomenderMentorModel
import tensorflow_recommenders as tfrs
import os

path = 'E:\\Projects\\tutorin-ml\\data_source\\Data_RecomenderSystem (1).xlsx'
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


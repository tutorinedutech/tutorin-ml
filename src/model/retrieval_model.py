import tensorflow_recommenders as tfrs
import tensorflow as tf

from typing import Dict, Text

class UserModel(tf.keras.Model):
    def __init__(self, similar_ds, unique_kriteria_mentor_user, embedding_dim=32):
        super().__init__()
        self.similar_ds = similar_ds
        self.unique_kriteria_mentor_user = unique_kriteria_mentor_user
        self.embedding_dim = embedding_dim
        max_tokens=10_000

        self.kriteria_mentor_user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=self.unique_kriteria_mentor_user, mask_token=None
            ),
            tf.keras.layers.Embedding(len(self.unique_kriteria_mentor_user)+1, self.embedding_dim)
        ])

        self.kriteria_mentor_user_vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens)
        self.kriteria_mentor_user_text_embedding = tf.keras.Sequential([
        self.kriteria_mentor_user_vectorizer,
            tf.keras.layers.Embedding(max_tokens, self.embedding_dim, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D()
        ])
        self.kriteria_mentor_user_vectorizer.adapt(self.similar_ds.map(lambda x: x['kriteria_mentor_user']))
    
    def call(self, inputs):
        return tf.concat([
            self.kriteria_mentor_user_embedding(inputs['kriteria_mentor_user']),
            self.kriteria_mentor_user_text_embedding(inputs['kriteria_mentor_user'])
        ], axis=1)
    
class MentorModel(tf.keras.Model):
    def __init__(self, unique_mentor_ids, embedding_dim=32):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.unique_mentor_ids = unique_mentor_ids
        self.mentor_id_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=self.unique_mentor_ids, mask_token=None),
            tf.keras.layers.Embedding(len(self.unique_mentor_ids)+1, embedding_dim)
        ])

    def call(self, inputs):
        return self.mentor_id_embedding(inputs)
        
class RecomenderMentorModel(tfrs.models.Model):
    def __init__(self, similar_ds, mentor_ds, unique_kriteria_mentor_user, unique_mentor_ids):
        super().__init__()
        self.query_model = tf.keras.Sequential([
            UserModel(similar_ds, unique_kriteria_mentor_user),
            tf.keras.layers.Dense(32)
        ])
        self.candidate_model = tf.keras.Sequential([
            MentorModel(unique_mentor_ids),
            tf.keras.layers.Dense(32)
        ])
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
            candidates=mentor_ds.batch(128).map(self.candidate_model),
            ),
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        query_embeddings = self.query_model({
            "kriteria_mentor_user": features["kriteria_mentor_user"],
        })
        mentor_embeddings = self.candidate_model(features['mentor_id'])

        return self.task(query_embeddings, mentor_embeddings)

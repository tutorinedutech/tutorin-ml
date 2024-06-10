import tensorflow_recommenders as tfrs
import tensorflow as tf

from typing import Dict, Text

class UserModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        max_tokens=10_000
        self.kriteria_mentor_user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_kriteria_mentor_user, mask_token=None
            ),
            tf.keras.layers.Embedding(len(unique_kriteria_mentor_user)+1, embedding_dim)
        ])

        self.kriteria_mentor_user_vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens)
        self.kriteria_mentor_user_text_embedding = tf.keras.Sequential([
        self.kriteria_mentor_user_vectorizer,
            tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D()
        ])
        self.kriteria_mentor_user_vectorizer.adapt(similar.map(lambda x: x['kriteria_mentor_user']))
    
    def call(self, inputs):
        return tf.concat([
            self.kriteria_mentor_user_embedding(inputs['kriteria_mentor_user']),
            self.kriteria_mentor_user_text_embedding(inputs['kriteria_mentor_user'])
        ], axis=1)
    
class MentorModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.mentor_id_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_mentor_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_mentor_ids)+1, 32)
        ])

    def call(self, inputs):
        return self.mentor_id_embedding(inputs)
        
class RecomenderMentorModel(tfrs.models.Model):
    def __init__(self):
        super().__init__()
        self.query_model = tf.keras.Sequential([
            UserModel(),
            tf.keras.layers.Dense(32)
        ])
        self.candidate_model = tf.keras.Sequential([
            MentorModel(),
            tf.keras.layers.Dense(32)
        ])
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
            candidates=mentor.batch(128).map(self.candidate_model),
            ),
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        query_embeddings = self.query_model({
            "user_id": features["user_id"],
            "kriteria_mentor_user": features["kriteria_mentor_user"],
        })
        mentor_embeddings = self.candidate_model(features['mentor_id'])

        return self.task(query_embeddings, mentor_embeddings)

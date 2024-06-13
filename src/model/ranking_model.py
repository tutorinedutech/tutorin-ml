import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import Dict, Text

class RankingModel(tf.keras.Model):
    def __init__(self, similar_ds, max_tokens, embedding_dimension=32):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.max_tokens = max_tokens
        self.similar_ds = similar_ds

        # Compute embeddings for users.
        self.kriteria_mentor_user_vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens)
        self.user_embeddings = tf.keras.Sequential([
            self.kriteria_mentor_user_vectorizer,
            tf.keras.layers.Embedding(self.max_tokens, self.embedding_dimension),
            tf.keras.layers.GlobalAveragePooling1D()
        ])
        self.kriteria_mentor_user_vectorizer.adapt(self.similar_ds.map(lambda x: x['kriteria_mentor_user']))

        # Compute embeddings for mentor.
        self.kriteria_mentor_vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens)
        self.mentor_embeddings = tf.keras.Sequential([
                self.kriteria_mentor_vectorizer,
                tf.keras.layers.Embedding(self.max_tokens, self.embedding_dimension),
                tf.keras.layers.GlobalAveragePooling1D()
        ])
        self.kriteria_mentor_vectorizer.adapt(self.similar_ds.map(lambda x: x['kriteria_mentor']))

        # Compute predictions.
        self.similarity = tf.keras.Sequential([
        # Learn multiple dense layers.
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        # Make rating predictions in the final layer.
        tf.keras.layers.Dense(1)
        ])

    def call(self, inputs):

        kriteria_mentor_user, kriteria_mentor = inputs

        user_embedding = self.user_embeddings(kriteria_mentor_user)
        mentor_embedding = self.mentor_embeddings(kriteria_mentor)

        return self.similarity(tf.concat([user_embedding, mentor_embedding], axis=1))
  
class MentorModel(tfrs.models.Model):

    def __init__(self, similar_ds, max_tokens, embedding_dimension=32):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel(similar_ds, max_tokens, embedding_dimension)
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss = tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
        return self.ranking_model(
            (features["kriteria_mentor_user"], features["kriteria_mentor"]))

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        labels = features.pop("similarity")

        rating_predictions = self(features)

        # The task computes the loss and the metrics.
        return self.task(labels=labels, predictions=rating_predictions)

import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import Dict, Text

class RankingModel(tf.keras.Model):
    def __init__(self, unique_kriteria_mentor_user, unique_mentor_id, embedding_dimension):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.unique_kriterial_mentor_user = unique_kriteria_mentor_user
        self.unique_mentor_id = unique_mentor_id

        # Compute embeddings for users.
        self.user_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
            vocabulary=self.unique_kriteria_mentor_user, mask_token=None),
            tf.keras.layers.Embedding(len(self.unique_kriteria_mentor_user) + 1, embedding_dimension)
        ])

        # Compute embeddings for movies.
        self.mentor_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=self.unique_mentor_id, mask_token=None),
                tf.keras.layers.Embedding(len(self.unique_mentor_id) + 1, embedding_dimension)
            ])

        # Compute predictions.
        self.similarity = tf.keras.Sequential([
        # Learn multiple dense layers.
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        # Make rating predictions in the final layer.
        tf.keras.layers.Dense(1)
        ])

def call(self, inputs):

    kriteria_mentor_user, mentor_id = inputs

    user_embedding = self.user_embeddings(kriteria_mentor_user)
    mentor_embedding = self.movie_embeddings(mentor_id)

    return self.similarity(tf.concat([user_embedding, mentor_embedding], axis=1))
  
class MovielensModel(tfrs.models.Model):

    def __init__(self):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel()
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss = tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
        return self.ranking_model(
            (features["kriteria_mentor_user"], features["mentor_ic"]))

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        labels = features.pop("similarity")

        rating_predictions = self(features)

        # The task computes the loss and the metrics.
        return self.task(labels=labels, predictions=rating_predictions)

import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import Dict, Text

class RankingModel(tf.keras.Model):
    def __init__(self, unique_kriteria_mentor_user, unique_kriteria_mentor, embedding_dimension=32):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.unique_kriteria_mentor_user = unique_kriteria_mentor_user
        self.unique_kriteria_mentor = unique_kriteria_mentor

        # Compute embeddings for users.
        self.user_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
            vocabulary=self.unique_kriteria_mentor_user, mask_token=None),
            tf.keras.layers.Embedding(len(self.unique_kriteria_mentor_user) + 1, self.embedding_dimension)
        ])

        # Compute embeddings for mentor.
        self.mentor_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=self.unique_kriteria_mentor, mask_token=None),
                tf.keras.layers.Embedding(len(self.unique_kriteria_mentor) + 1, self.embedding_dimension)
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

        kriteria_mentor_user, kriteria_mentor = inputs

        user_embedding = self.user_embeddings(kriteria_mentor_user)
        mentor_embedding = self.mentor_embeddings(kriteria_mentor)

        return self.similarity(tf.concat([user_embedding, mentor_embedding], axis=1))
  
class MentorModel(tfrs.models.Model):

    def __init__(self, unique_kriteria_mentor_user, unique_kriteria_mentor, embedding_dimension=32):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel(unique_kriteria_mentor_user, unique_kriteria_mentor, embedding_dimension)
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

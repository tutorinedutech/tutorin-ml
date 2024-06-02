import keras
import keras_nlp
import tensorflow as tf

class SiameseModel:
    def __init__(self, encoder_preset='roberta_base_en', **kwargs):
        super().__init__(**kwargs)
        self.preprocessor = keras_nlp.models.RobertaPreprocessor.from_preset(encoder_preset)
        self.backbone = keras_nlp.models.RobertaBackbone.from_preset(encoder_preset)

          # Define the inputs and preprocessing
        self.sentence_input = keras.Input(shape=(1,), dtype='string', name='sentence')
        self.preprocessed_input = self.preprocessor(self.sentence_input)
        self.backbone_output = self.backbone(self.preprocessed_input)
        
        # Define the pooling and normalization layers
        self.pooling_layer = tf.keras.layers.GlobalAveragePooling1D(name='pooling_layer')
        self.normalization_layer = tf.keras.layers.UnitNormalization(axis=1)
        
        # Define the full encoder model
        self.encoder = tf.keras.models.Model(
            inputs=self.sentence_input,
            outputs=self.normalization_layer(self.pooling_layer(self.backbone_output, self.preprocessed_input['padding_mask']))
        )
        
        # Define the inputs for the Siamese model
        inputs = keras.Input(shape=(2,), dtype="string", name="sentences")
        sen1, sen2 = tf.keras.ops.split(inputs, 2, axis=1)
        
        # Get the embeddings
        u = self.encoder(sen1)
        v = self.encoder(sen2)
        
        # Compute the cosine similarity
        cosine_similarity_scores = tf.keras.layers.Dot(axes=1, normalize=True)([u, v])
        
        self.model = tf.keras.models.Model(inputs=inputs, outputs=cosine_similarity_scores)
        
    def call(self, inputs):
        return self.model(inputs)

    def get_encoder(self):
        return self.encoder

if __name__ == "__main__":
    # Example of creating and using the RegressionSiamese model
    siamese_model = SiameseModel()

    #Print model summary
    siamese_model.model.summary()
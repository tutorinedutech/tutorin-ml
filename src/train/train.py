from tensorflow.keras.models import load_model

import pandas as pd

from src.model.siamese_lstm import SiameneLSTM
from src.data.pre_processing import word_embed_meta_data
from operator import itemgetter

#initialized required parameters for LSTM network...
EMBEDDING_DIM = 50
MAX_SEQUENCE_LENGTH = 10
VALIDATION_SPLIT = 0.1
RATE_DROP_LSTM = 0.17
RATE_DROP_DENSE = 0.25
NUMBER_LSTM = 50
NUMBER_DENSE_UNITS = 50
ACTIVATION_FUNCTION = 'relu'

def train_dataset_model(df):
    df['Prompt'] = df['Prompt'].astype(str)
    df['combined'] = df['combined'].astype(str)
    
    prompt = df['Prompt'].tolist()
    kriteria = df['combined'].tolist()
    
    
    # creating word embedding meta data for word embedding 
    embedding_matrix = word_embed_meta_data(prompt+kriteria,  EMBEDDING_DIM)
    embedding_meta_data = {'embedding_matrix': embedding_matrix}
    print("----------created word embedding meta data-----------")
    
    #SiameneBiLSTM is a class for  Long short Term Memory networks
    siamese = SiameneLSTM(EMBEDDING_DIM ,MAX_SEQUENCE_LENGTH, NUMBER_LSTM, NUMBER_DENSE_UNITS, RATE_DROP_LSTM, RATE_DROP_DENSE, ACTIVATION_FUNCTION, VALIDATION_SPLIT)
    model_path = siamese.train_model(df, embedding_meta_data, model_save_directory='./')
    
    #load the train data in model...
    model = load_model(model_path)
    print("----------model trained-----------")
    return model

if __name__ == '__main__':
    df = pd.read_csv('E:\\Projects\\tutorin-ml\\data_source\\processed\\clean_data.csv')

    train_model = train_dataset_model(df)
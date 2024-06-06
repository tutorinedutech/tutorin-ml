from tensorflow.keras.models import load_model
from src.data.pre_processing import word_embed_meta_data, create_test_data, create_train_dev
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

EMBEDDING_DIM = 50

df = pd.read_csv('E:\\Projects\\tutorin-ml\\data_source\\processed\\clean_data.csv')

df['Prompt'] = df['Prompt'].astype(str)
df['combined'] = df['combined'].astype(str)
    
question1 = df['Prompt']
question2 = df['combined']
label = df['Skor Label']
    
## creating questions pairs
#questions_pair = [(x1, x2) for x1, x2 in zip(question1, question2)]
#print("----------created questions pairs-----------")



tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(question1 + question2)

new_model = load_model('E:\Projects\\tutorin-ml\checkpoints\\1717653513\lstm_50_50_0.17_0.25.keras')


seq = tokenizer.texts_to_sequences(['I love you'])
print(seq)

